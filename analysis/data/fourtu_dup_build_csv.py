"""Build docs/datasets/fourtu_duplicate_radargrams.csv from the scan (no GT depth values).

Reads data/analysis/fourtu_dup/ (git-ignored) fourtu_dup_scan.json and
fourtu_dup_pairs.json; writes the committed csv (utf-8-sig); the log is stdout.
Run order: fourtu_dup_scan -> fourtu_dup_analyze -> fourtu_dup_extra ->
fourtu_dup_build_csv. Example (from the project root, PowerShell):
    $log = "data/analysis/fourtu_dup/fourtu_dup_build_csv.log"
    conda run -n gpr-dev python -I analysis/data/fourtu_dup_build_csv.py *> $log
"""
import json
from pathlib import Path

import pandas as pd

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = load_dataset_paths().fourtu
A = PROJECT_ROOT / "data" / "analysis" / "fourtu_dup"
D = json.loads((A / "fourtu_dup_scan.json").read_text(encoding="utf-8"))
byk = {f"{r['activity']}/{r['path']}": r for r in D["files"]}
pairs = json.loads((A / "fourtu_dup_pairs.json").read_text())["pairs"]
am = pd.read_csv(ROOT / "01_메타데이터" / "activities_metadata_ko.csv", dtype={"activity_id": str}).set_index("activity_id")
gt = pd.read_csv(ROOT / "01_메타데이터" / "utilities_groundtruth.csv", usecols=["activity_id"], dtype={"activity_id": str})
gtn = gt.groupby("activity_id").size()


def tstr(r):
    y, d, h, m, s = r["sgy_time_le"]
    return f"{y}-d{d:03d} {h:02d}:{m:02d}:{s:02d}"


def sort_key(p):
    a, b = p
    def k(x):
        act, path = x.split("/")
        s, n = act.split(".")
        return (int(s), int(n), int(path[4:]))
    return (k(a), k(b))


# order each pair so that a < b (site, activity number, path number)
def ordered(p):
    a, b = p
    return (a, b) if sort_key((a, b))[0] <= sort_key((b, a))[0] else (b, a)


pairs = sorted((ordered(p) for p in pairs), key=lambda p: sort_key(p)[0])
rows = []
for i, (a, b) in enumerate(pairs, 1):
    ra, rb = byk[a], byk[b]
    aa, ab = ra["activity"], rb["activity"]
    within = aa == ab
    same_sgy = ra["sgy_sha"] == rb["sgy_sha"]
    ev = []
    if same_sgy:
        ev.append("원본 SEG-Y 멤버 SHA-256·ZIP CRC32 동일")
    else:
        ev.append("SEG-Y 전체 SHA-256은 다름(텍스트 헤더 40행만 다름), 트레이스 데이터·트레이스 헤더 SHA-256 동일")
    ev.append(f"서로 다른 ZIP 멤버 경로({ra['source_member']} / {rb['source_member']})")
    ev.append(f"첫 트레이스 헤더 취득 시각 동일({tstr(ra)})")
    if within:
        cls = "(i) 원 데이터셋 중복 - 활동 내 사본"
        sub = "i-a"
        ev.append("같은 활동 안 다른 측선 번호에 같은 파일(ZIP 멤버 수정시각 +98초: 2020-04-03 12:56:34 vs 12:58:12)")
        ev.append("정답 행 없는 활동(09.5)")
    else:
        sa, sb = len(ra["activity"]), 0
        # whole-set vs boundary single line
        n_a = sum(1 for k in byk if byk[k]["activity"] == aa)
        n_b = sum(1 for k in byk if byk[k]["activity"] == ab)
        npairs_act = sum(1 for p in pairs if {byk[p[0]]["activity"], byk[p[1]]["activity"]} == {aa, ab})
        if npairs_act >= 3 or (n_a == npairs_act and n_b == npairs_act):
            cls = "(i) 원 데이터셋 중복 - 활동 간 측선 세트 복제"
            sub = "i-b"
        else:
            cls = "(i) 원 데이터셋 중복 - 활동 간 경계 단일 측선(소속 활동 판단 불가)"
            sub = "i-c"
        ev.append("두 활동의 조건 메타(지표 유형·날씨 등)·정답표 행 수가 서로 달라 한 번의 측정이 두 활동 모두에 속할 수 없거나 한쪽 배정이 틀림")
    rows.append({
        "pair_id": i, "sha256_npy": ra["npy_sha"],
        "file_a": f"05_radargrams_npy/{aa}/{ra['path']}.npy", "file_b": f"05_radargrams_npy/{ab}/{rb['path']}.npy",
        "site": int(am.loc[aa, "site"]), "scope": "within_activity" if within else "across_activities",
        "activity_a": aa, "path_a": ra["path"], "activity_b": ab, "path_b": rb["path"], "n_traces": ra["shape"][1],
        "gt_rows_a": int(gtn.get(aa, 0)), "gt_rows_b": int(gtn.get(ab, 0)),
        "class_code": sub, "classification": cls, "evidence": "; ".join(ev),
    })
df = pd.DataFrame(rows)
out = PROJECT_ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.groupby(["class_code", "scope"]).size())
print(df.groupby(["activity_a", "activity_b"]).size())
print(len(df), df["sha256_npy"].nunique())
