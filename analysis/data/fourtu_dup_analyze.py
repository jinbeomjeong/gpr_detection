"""Analyze fourtu_dup_scan.json: duplicate groups, SEG-Y comparison, metadata comparison.
Reads only row counts of the ground-truth table (activity_id column); never depth values.

Reads and writes data/analysis/fourtu_dup/ (git-ignored): fourtu_dup_scan.json in,
fourtu_dup_pairs_detail.csv and fourtu_dup_pairs.json out; the log is stdout.
Run order: fourtu_dup_scan -> fourtu_dup_analyze -> fourtu_dup_extra ->
fourtu_dup_build_csv. Example (from the project root, PowerShell):
    $log = "data/analysis/fourtu_dup/fourtu_dup_analyze.log"
    conda run -n gpr-dev python -I analysis/data/fourtu_dup_analyze.py *> $log
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "fourtu_dup"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT = load_dataset_paths().fourtu
D = json.loads((OUT_DIR / "fourtu_dup_scan.json").read_text(encoding="utf-8"))
F = D["files"]
for r in F:
    r["key"] = f"{r['activity']}/{r['path']}"
byk = {r["key"]: r for r in F}

print("files", len(F))
# full-file duplicate groups on npy bytes
g = defaultdict(list)
for r in F:
    g[r["npy_sha"]].append(r["key"])
groups = [v for v in g.values() if len(v) > 1]
print("npy-hash dup groups", len(groups), "files in groups", sum(len(v) for v in groups))
print("sizes", sorted(set(len(v) for v in groups)))
# pairs
pairs = []
for v in groups:
    for a, b in combinations(sorted(v), 2):
        pairs.append((a, b))
print("pairs", len(pairs))

# array-level dups (ignores npy header differences) and flipped
g2 = defaultdict(list)
for r in F:
    g2[r["arr_sha"]].append(r["key"])
print("arr-hash dup groups", sum(1 for v in g2.values() if len(v) > 1))
flip = 0
arrset = {r["arr_sha"]: r["key"] for r in F}
for r in F:
    k = arrset.get(r["arr_flip_sha"])
    if k and k != r["key"]:
        flip += 1
        print("FLIP", r["key"], k)
print("flip matches", flip)

# sgy-level dups
g3 = defaultdict(list)
for r in F:
    g3[r["sgy_sha"]].append(r["key"])
sgy_groups = [v for v in g3.values() if len(v) > 1]
print("sgy-hash dup groups", len(sgy_groups), "files", sum(len(v) for v in sgy_groups))
g4 = defaultdict(list)
for r in F:
    g4[r["sgy_data_sha"]].append(r["key"])
print("sgy trace-data dup groups", sum(1 for v in g4.values() if len(v) > 1))

# dup of other kinds: png_raw
g5 = defaultdict(list)
for r in F:
    g5[r["png_raw_sha"]].append(r["key"])
print("png_raw dup groups", sum(1 for v in g5.values() if len(v) > 1))

# partial overlaps via column hashes (non-zero columns)
col = defaultdict(set)
for r in F:
    for i, h in enumerate(r["sgy_col_hashes"]):
        col[h].add((r["key"], i))
shared = defaultdict(int)
for h, s in col.items():
    ks = sorted({k for k, _ in s})
    if 1 < len(ks) <= 6:
        for a, b in combinations(ks, 2):
            shared[(a, b)] += 1
pair_set = set(pairs)
partial = [(a, b, n) for (a, b), n in shared.items() if (a, b) not in pair_set and n >= 5]
print("partial overlap pairs (>=5 shared traces, not full dup) [raw, may include blank traces]", len(partial))
print("sgy-different/npy-same pairs:", [(a, b) for a, b in pairs if byk[a]["sgy_sha"] != byk[b]["sgy_sha"]])

# per-pair detail
rows = []
for a, b in pairs:
    ra, rb = byk[a], byk[b]
    rows.append({
        "a": a, "b": b, "npy_sha": ra["npy_sha"],
        "same_sgy": ra["sgy_sha"] == rb["sgy_sha"],
        "same_sgy_data": ra["sgy_data_sha"] == rb["sgy_data_sha"],
        "same_sgy_hdrs": ra["sgy_hdrs_sha"] == rb["sgy_hdrs_sha"],
        "same_text": ra["sgy_text_sha"] == rb["sgy_text_sha"],
        "same_bin": ra["sgy_bin_sha"] == rb["sgy_bin_sha"],
        "same_hdr0": ra["sgy_hdr0_sha"] == rb["sgy_hdr0_sha"],
        "same_zipmember": ra["source_member"] == rb["source_member"],
        "same_zip": ra["source_zip"] == rb["source_zip"],
        "crc_a": ra["zip_crc"], "crc_b": rb["zip_crc"],
        "size_a": ra["zip_size"], "size_b": rb["zip_size"],
        "dt_a": ra["zip_dt"], "dt_b": rb["zip_dt"],
        "time_a": ra["sgy_time_be"], "time_b": rb["sgy_time_be"],
        "tle_a": ra["sgy_time_le"], "tle_b": rb["sgy_time_le"],
        "ntr": ra["shape"][1],
        "png_same": ra["png_raw_sha"] == rb["png_raw_sha"],
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 50)
print(df[["a", "b", "ntr", "same_sgy", "same_sgy_data", "same_sgy_hdrs", "same_text", "same_bin", "same_hdr0", "same_zip", "dt_a", "dt_b"]].to_string())
print(df[["a", "b", "time_a", "time_b", "tle_a", "tle_b"]].head(10).to_string())
df.to_csv(OUT_DIR / "fourtu_dup_pairs_detail.csv", index=False, encoding="utf-8-sig")

# metadata comparison per activity pair
am = pd.read_csv(ROOT / "01_메타데이터" / "activities_metadata_ko.csv", dtype={"activity_id": str})
am = am.set_index("activity_id")
gt = pd.read_csv(ROOT / "01_메타데이터" / "utilities_groundtruth.csv", usecols=["activity_id"], dtype={"activity_id": str})
gtn = gt.groupby("activity_id").size()
acts = sorted({k.split("/")[0] for p in pairs for k in p})
print("GT rows per activity (counts only):")
for a in acts:
    print(" ", a, int(gtn.get(a, 0)), "site", am.loc[a, "site"], "ncols", am.shape[1])
actpairs = sorted({tuple(sorted((a.split("/")[0], b.split("/")[0]))) for a, b in pairs if a.split("/")[0] != b.split("/")[0]})
print("activity pairs", actpairs)
for a, b in actpairs:
    ra, rb = am.loc[a], am.loc[b]
    diff = [c for c in am.columns if str(ra[c]) != str(rb[c]) and not c.endswith("(원문)")]
    print(a, b, "GT rows", int(gtn.get(a, 0)), int(gtn.get(b, 0)), "meta diff cols:", {c: (ra[c], rb[c]) for c in diff if c not in ("activity_id", "activity_no")})

# index metadata comparison per pair
for a, b in pairs:
    ra, rb = byk[a], byk[b]
    ma = json.loads((ROOT / "05_radargrams_npy" / ra["activity"] / f"{ra['path']}.json").read_text(encoding="utf-8"))
    mb = json.loads((ROOT / "05_radargrams_npy" / rb["activity"] / f"{rb['path']}.json").read_text(encoding="utf-8"))
    d = {k: (ma[k], mb[k]) for k in ma if ma[k] != mb[k] and k not in ("activity_id", "path", "source_member", "source_zip")}
    if d:
        print("META DIFF", a, b, d)

# zip png (ground truth / survey map) hashes of activity pairs
zp = D["zip_pngs"]
def png_for(act, kind):
    for fn, v in zp.items():
        if f"/{act}/" in fn.replace("\\", "/") or fn.replace("\\", "/").split("/")[-1].startswith(f"{act}_"):
            if kind in fn:
                return v["sha"][:10], fn
    return None
for a, b in actpairs:
    for kind in ("ground-truth", "survey_map"):
        pa, pb = png_for(a, kind), png_for(b, kind)
        print("PNG", kind, a, b, pa and pa[0], pb and pb[0], "same" if pa and pb and pa[0] == pb[0] else "diff")
# within-activity pairs 09.5
json.dump({"pairs": pairs, "partial": partial}, open(OUT_DIR / "fourtu_dup_pairs.json", "w"))
