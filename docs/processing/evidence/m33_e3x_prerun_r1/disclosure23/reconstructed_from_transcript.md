# H6 공개 23 증거: 1회차 injected 결과 폴더 열람 기록 (transcript 전사)

**재실행 아님 · transcript 전사.** 아래 스크립트 본문·명령·stdout은 python-expert 서브에이전트
transcript에서 그대로 옮겼다. 원본 스크립트 파일(`defect_names.py`)과 stdout 파일은 디스크에 남아
있지 않다. scratchpad 폴더 `python-e3xdiag\`는 2026-11-20 15:25:31 KST(06:25:31Z) 정리 명령으로 지웠다.
이 전사를 위해 스크립트를 다시 돌리지 않았다.

- 출처 transcript: `C:\Users\jinbeom\.claude\projects\C--workspace-gpr-detection\8552c4b2-9d5c-455e-947e-0033764e1def\subagents\agent-a69e5d6b43ae10991.jsonl`
  - 전사 시점(2026-11-20 16:53 KST 무렵) 그 파일의 sha256은 `0e20ee93b3c8d5be865312b51844f7b7445eaf266bd373d6cb3e2ab15553aa80`(270줄)이다.
  - 이 sha는 전사 작업 중에 쟀다. transcript는 계속 덧붙여지는 파일이라 이후 sha는 달라진다. 50–60행 내용은 바뀌지 않는다.
- 같은 폴더의 `transcript_excerpt_lines_50_52_54_55_59_60.jsonl`은 위 transcript의 50·52·54·55·59·60행 원문 사본(sed 추출, 가공 없음)이다.

## 1. 결과 폴더 파일 이름 목록 (transcript 50·52행)

- 시각: 명령 2026-11-20T06:24:03.689Z(15:24:03 KST), 결과 06:24:04.765Z
- 명령 (Bash 도구):

```
ls "C:/workspace/gpr_detection/runs/eval/m33/e3x/injected/" | head -30; ls "C:/workspace/gpr_detection/runs/eval/m33/e3x/injected/" | wc -l
```

- stdout:

```
band_records.csv
hyperbola_records.csv
parts
realizations.csv
rejections.csv
stage_settings.json
summary.json
truths.csv
8
```

## 2. 스크립트 작성 (transcript 54·55행)

- 시각: Write 2026-11-20T06:24:11.524Z(15:24:11 KST)
- 경로: `C:\Users\jinbeom\AppData\Local\Temp\claude\8d33c416c891208f\8552c4b2-9d5c-455e-947e-0033764e1def\scratchpad\python-e3xdiag\defect_names.py`
- 본문:

```python
"""Print only the names of failed defect checks (blind: no counts, no other keys).

Reads the ``defects`` key of ``summary.json`` and, per part file, only its
``defects`` and ``line_id`` keys. Counts are not printed.
"""

import json
import sys
from pathlib import Path


def failed_names(defects: dict) -> list[str]:
    return sorted(name for name, count in defects.items() if count)


def main() -> None:
    stage_dir = Path(sys.argv[1])
    summary = json.loads((stage_dir / "summary.json").read_text(encoding="utf-8"))
    defects = summary["defects"]
    print("summary.defects keys recorded:", sorted(defects))
    print("summary.defects failed names:", failed_names(defects))
    for part_path in sorted((stage_dir / "parts").glob("*.json")):
        part = json.loads(part_path.read_text(encoding="utf-8"))
        names = failed_names(part["defects"])
        if names:
            print(f"part {part_path.name} line_id={part['line_id']}: {names}")


if __name__ == "__main__":
    main()
```

주의: 이 스크립트는 `summary.json`과 `parts\*.json` 각 파일 **전체를 메모리로 로드**했다. 출력한 것은
`summary.json`의 `defects` 키 이름(전부)과 값이 0이 아닌 키 이름, 그리고 part 파일의 `defects`(0이 아닌 키 이름)와
`line_id`뿐이다. 건수와 다른 키는 출력하지 않았다.

## 3. 실행 (transcript 59·60행)

- 시각: 명령 2026-11-20T06:24:15.752Z(15:24:15 KST), 결과 06:24:29.975Z(15:24:29 KST)
- 명령 (PowerShell 도구):

```
conda run -n gpr-dev python -I -B "C:\Users\jinbeom\AppData\Local\Temp\claude\8d33c416c891208f\8552c4b2-9d5c-455e-947e-0033764e1def\scratchpad\python-e3xdiag\defect_names.py" "C:\workspace\gpr_detection\runs\eval\m33\e3x\injected"
```

- stdout (transcript 원문의 `\r\n`을 줄바꿈으로 바꿨다):

```
summary.defects keys recorded: ['nms_time_limit_warning', 'off_differs_from_on', 'r_edge_differs_from_plan', 'shift_mismatch', 'texture_check', 'tile_with_300_operating_boxes', 'zero_test']
summary.defects failed names: ['texture_check']
part 4tu__010.18__Path2.json line_id=010.18/Path2: ['texture_check']
part 4tu__011.16__Profile_6453.json line_id=011.16/Profile_6453: ['texture_check']
part gz_ids__pipe__1030_181230AB__LID10001.json line_id=pipe/1030_181230AB/LID10001: ['texture_check']
```

## 4. 그 밖의 열람
- 이 두 명령 말고는 injected 결과 폴더의 파일을 열거나 읽지 않았다.
- 그 뒤의 질감 진단(3측선·precheck 2측선, 15:27 무렵)과 전체 155측선 질감 사전 검사(15:33–15:37)는 injected 결과 폴더를 열지 않았다. 둘이 읽은 것은 plan 표, pool, 응답 라이브러리, 원 측선이다.
