"""Read-only review of the M3-2 stage A lists (no input file is written).

Checks: AIHub reference vs trainpool/test seal, session/gain composition; 4TU list duplicates (byte hash),
site spread, K bundle disjointness; IDS equipment/frequency distribution and MALA exclusion; duplicates and
missing paths; re-derivation of the three lists with seed 20261114 (copy of the selection logic) and compare.
Output: JSON to the path given in argv[1].
Run: conda run -n gpr-dev python -I analysis/data/m32_stageA_lists_review.py <out.json>
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\workspace\gpr_detection")
PROC = ROOT / "analysis" / "processing"
GPR = Path(r"C:\workspace\GPR_Data")
FOURTU = next(GPR.glob("4TU_*")) / "05_radargrams_npy"
GZ = next(GPR.glob("광저우대_*"))
SEED = 20261114


def rows(p):
    with open(p, encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def proportional_pick(groups, n_total, rng):
    keys = sorted(groups)
    size = sum(len(groups[k]) for k in keys)
    if size <= n_total:
        return [x for k in keys for x in groups[k]]
    quota = {k: n_total * len(groups[k]) / size for k in keys}
    alloc = {k: int(math.floor(quota[k])) for k in keys}
    rest = n_total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    out = []
    for k in keys:
        m = groups[k]
        idx = rng.choice(len(m), size=min(alloc[k], len(m)), replace=False)
        out.extend(m[i] for i in sorted(idx))
    return out


res = {}
# ---------------- AIHub
split = rows(ROOT / "data" / "splits" / "aihub_yolo_v1.csv")
ref = rows(PROC / "m32_stageA_aihub_ref.csv")
by_file = {r["file_name"]: r for r in split}
by_key = defaultdict(dict)
for r in split:
    by_key[r["crop_key"]][r["gain"]] = r
a = {}
a["n_ref"] = len(ref)
a["class_counts"] = Counter(r["class"] for r in ref)
a["dup_crop_keys"] = len(ref) - len({r["crop_key"] for r in ref})
subsets, sess, fold, equip, miss_in_split, miss_on_disk, wrong_class, wrong_key = [], [], [], [], 0, 0, 0, 0
imgdir = ROOT / "data" / "aihub_yolo_v1" / "images"
gaincount = Counter()
sess_cls = defaultdict(Counter)
test_sessions = {r["session"] for r in split if r["subset"] == "test"}
tp_sessions = {r["session"] for r in split if r["subset"] == "trainpool"}
a["session_overlap_test_trainpool_all"] = len(test_sessions & tp_sessions)
for r in ref:
    for g in ("1.0", "0.5", "0.25"):
        fn = r[f"g{g}"]
        if not fn:
            gaincount[f"missing_{g}"] += 1
            continue
        gaincount[g] += 1
        s = by_file.get(fn)
        if s is None:
            miss_in_split += 1
            continue
        subsets.append(s["subset"])
        if s["crop_key"] != r["crop_key"]:
            wrong_key += 1
        if s["gain"] != g:
            wrong_key += 1
        if not (imgdir / fn).exists():
            miss_on_disk += 1
    s1 = by_file[r["g1.0"]]
    sess.append(s1["session"])
    fold.append(s1["fold"])
    equip.append(r["crop_key"].split("_")[0])
    sess_cls[r["class"]][s1["session"]] += 1
    if s1["primary_class"] != r["class"]:
        wrong_class += 1
a["subsets"] = Counter(subsets)
a["missing_in_split"] = miss_in_split
a["missing_on_disk"] = miss_on_disk
a["wrong_class_or_key"] = [wrong_class, wrong_key]
a["gain_counts"] = gaincount
a["equipment"] = Counter(equip)
a["fold"] = Counter(fold)
a["n_sessions"] = len(set(sess))
a["session_in_test"] = len(set(sess) & test_sessions)
a["per_class_sessions"] = {c: len(v) for c, v in sess_cls.items()}
a["per_class_top_sessions"] = {c: v.most_common(5) for c, v in sess_cls.items()}
a["per_class_max_session_share"] = {c: max(v.values()) / sum(v.values()) for c, v in sess_cls.items()}
# pool availability per class (gain 1.0 trainpool)
pool = [r for r in split if r["subset"] == "trainpool"]
avail = Counter()
avail_sess = defaultdict(set)
avail_equip = defaultdict(Counter)
for r in pool:
    if r["gain"] == "1.0":
        avail[r["primary_class"]] += 1
        avail_sess[r["primary_class"]].add(r["session"])
        avail_equip[r["primary_class"]][r["crop_key"].split("_")[0]] += 1
a["pool_g1_counts"] = avail
a["pool_sessions"] = {c: len(v) for c, v in avail_sess.items()}
a["pool_equip"] = {c: dict(v) for c, v in avail_equip.items() if c.startswith("pipe")}
# selection restricted to NM? how many non-NM pool crops of pipe classes exist
a["pool_pipe_non_NM"] = {c: sum(v for e, v in avail_equip[c].items() if e != "NM") for c in ("pipe_hyperbola", "pipe_band")}
# pool NM sessions per class
nm_sess = defaultdict(set)
for r in pool:
    if r["gain"] == "1.0" and r["primary_class"].startswith("pipe") and r["crop_key"].startswith("NM"):
        nm_sess[r["primary_class"]].add(r["session"])
a["pool_NM_sessions"] = {c: len(v) for c, v in nm_sess.items()}
# gain availability for all trainpool NM pipe keys
a["pool_keys_missing_g05"] = sum(1 for k, d in by_key.items() if k.startswith("NM") and "1.0" in d and d["1.0"]["subset"] == "trainpool" and "0.5" not in d)
# overlapping crops among selected on same line
lines = defaultdict(list)
for r in ref:
    k = r["crop_key"]
    base, xr = k.rsplit("_x(", 1)
    x0, x1 = (int(v) for v in xr.rstrip(")").split("-"))
    lines[base].append((x0, x1, r["class"]))
ov = 0
for base, lst in lines.items():
    lst.sort()
    for i in range(len(lst) - 1):
        if lst[i + 1][0] < lst[i][1]:
            ov += 1
a["n_lines"] = len(lines)
a["overlapping_adjacent_pairs"] = ov
a["lines_both_classes"] = sum(1 for l in lines.values() if len({c for *_, c in l}) > 1)
res["aihub"] = a

# ---------------- 4TU
l4 = rows(PROC / "m32_stageA_lines_4tu.csv")
dups = rows(ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv")
d3 = rows(ROOT / "data" / "processing" / "d3" / "lines.csv")
lines4 = rows(ROOT / "docs" / "processing" / "fourtu_lines_v1.csv")
f = {}
f["n"] = len(l4)
f["dup_lines"] = len(l4) - len({r["line"] for r in l4})
f["source"] = Counter(r["source"] for r in l4)
f["site"] = dict(sorted(Counter(int(r["site"]) for r in l4).items()))
f["dt"] = Counter(r["dt_ns"] for r in l4)
allsite = Counter(int(r["site"]) for r in lines4 if r["npy_exists"] == "True")
f["all_site"] = dict(sorted(allsite.items()))
f["sites_missing"] = [s for s in range(1, 14) if s not in f["site"]]
dropped = {f"{r['activity_b']}/{r['path_b']}" for r in dups}
dropped_a = {f"{r['activity_a']}/{r['path_a']}" for r in dups}
inl = {r["line"] for r in l4}
f["file_b_in_list"] = sorted(inl & dropped)
f["n_dup_pairs"] = len(dups)
f["file_a_in_list"] = len(inl & dropped_a)
kset = {f"{r['activity_id']}/{r['path']}" for r in d3 if "K" in r["bundles"]}
f["K_lines"] = len(kset)
f["K_in_list"] = sorted(inl & kset)
f["K_in_list_count"] = len(inl & kset)
bund = {f"{r['activity_id']}/{r['path']}": r["bundles"] for r in d3}
f["list_bundles"] = Counter(bund.get(x, "-") for x in inl)
f["SLX_total_in_d3_not_K"] = sum(1 for k, b in bund.items() if any(c in b for c in "SLX") and "K" not in b)
f["SLX_lines_in_list_vs_d3"] = [sum(1 for r in l4 if r["source"] == "SLX"), f["SLX_total_in_d3_not_K"]]
f["SK_line"] = [k for k, b in bund.items() if "K" in b and "S" in b]
f["missing_files"] = [r["line"] for r in l4 if not (FOURTU / (r["line"].replace("/", "\\") + ".npy")).exists()]
meta = {f"{r['activity_id']}/{r['path']}": r for r in lines4}
f["meta_mismatch"] = [r["line"] for r in l4 if r["line"] in meta and (meta[r["line"]]["site"] != r["site"] or meta[r["line"]]["dt_ns"] != r["dt_ns"])]
# byte hash
hs = defaultdict(list)
for r in l4:
    p = FOURTU / (r["line"].replace("/", "\\") + ".npy")
    if p.exists():
        hs[sha(p)].append(r["line"])
f["byte_identical_within_list"] = [v for v in hs.values() if len(v) > 1]
# site proportional vs list
tot = sum(allsite.values())
f["site_expected_share_vs_actual"] = {s: [round(allsite[s] / tot * 150, 1), f["site"].get(s, 0)] for s in sorted(allsite)}
# dt strata
alldt = Counter(r["dt_ns"] for r in lines4 if r["npy_exists"] == "True")
f["all_dt"] = alldt
res["fourtu"] = f

# ---------------- IDS
li = rows(PROC / "m32_stageA_lines_ids.csv")
idx = rows(GZ / "01_메타데이터" / "radargrams_index.csv")
im = {r["file_id"]: r for r in idx}
g = {}
g["n"] = len(li)
g["dup"] = len(li) - len({r["line"] for r in li})
g["not_in_index"] = [r["line"] for r in li if r["line"] not in im]
g["format"] = Counter(im[r["line"]]["format"] for r in li if r["line"] in im)
g["target"] = Counter(im[r["line"]]["target"] for r in li if r["line"] in im)
g["antenna"] = Counter(r["antenna_MHz"] for r in li)
g["tw"] = Counter(str(round(float(r["time_window_ns"]))) for r in li)
g["mission_zone_folder"] = Counter(im[r["line"]]["mission"] for r in li if r["line"] in im)
g["zone"] = Counter(im[r["line"]]["zone"] for r in li if r["line"] in im)
g["n_zones"] = len(g["zone"])
g["index_format_all"] = Counter(r["format"] for r in idx)
g["index_target_format"] = Counter(r["target"] + "|" + r["format"] for r in idx)
pipe_ids = [r for r in idx if r["target"] == "pipe" and r["format"] != "MALA .rd3"]
g["eligible"] = len(pipe_ids)
g["eligible_antenna"] = Counter(r["antenna_MHz"] for r in pipe_ids)
g["eligible_cells"] = Counter(r["antenna_MHz"] + "|" + str(round(float(r["time_window_ns"]))) for r in pipe_ids)
g["list_cells"] = Counter(r["antenna_MHz"] + "|" + str(round(float(r["time_window_ns"]))) for r in li)
g["mala_rows"] = Counter(r["antenna_MHz"] for r in idx if r["format"] == "MALA .rd3")
g["mala_in_list"] = [r["line"] for r in li if im.get(r["line"], {}).get("format") == "MALA .rd3"]
g["npy_missing"] = [r["line"] for r in li if not (GZ / "05_radargrams_npy" / (r["line"].replace("/", "\\") + ".npy")).exists()]
g["per_zone_max"] = max(g["zone"].values()) if g["zone"] else 0
g["nonpipe_in_list"] = sum(1 for r in li if im.get(r["line"], {}).get("target") != "pipe")
res["ids"] = g

# ---------------- reproduction (copy of build_lists, no write)
rng = np.random.default_rng(SEED)
rep = {}
ref2 = []
for cls in ("pipe_hyperbola", "pipe_band"):
    keys = sorted({r["crop_key"] for r in pool if r["primary_class"] == cls and r["gain"] == "1.0"})
    pick = rng.choice(len(keys), size=min(300, len(keys)), replace=False)
    for i in sorted(pick):
        ref2.append((cls, keys[i]))
rep["aihub_same"] = ref2 == [(r["class"], r["crop_key"]) for r in ref]
dropped_t = {(r["activity_b"], r["path_b"]) for r in dups}
d3m = {(r["activity_id"], r["path"]): r["bundles"] for r in d3}
k_lines = {k for k, b in d3m.items() if "K" in b}
all4 = [r for r in lines4 if r.get("npy_exists", "True") == "True" and (r["activity_id"], r["path"]) not in dropped_t and (r["activity_id"], r["path"]) not in k_lines]
core = [r for r in all4 if any(x in d3m.get((r["activity_id"], r["path"]), "") for x in ("S", "L", "X"))]
ck = {(r["activity_id"], r["path"]) for r in core}
restl = [r for r in all4 if (r["activity_id"], r["path"]) not in ck]
groups = defaultdict(list)
for r in restl:
    groups[(r["site"], r["dt_ns"])].append(r)
fill = proportional_pick(groups, 150 - len(core), rng)
l4b = [f"{r['activity_id']}/{r['path']}" for r in core + fill]
rep["fourtu_same"] = l4b == [r["line"] for r in l4]
rep["core_n"] = len(core)
rep["rest_pool_n"] = len(restl)
rep["fourtu_strata"] = len(groups)
gz2 = defaultdict(list)
for r in pipe_ids:
    gz2[(r["antenna_MHz"], str(round(float(r["time_window_ns"]))))].append(r)
l2 = [r["file_id"] for r in proportional_pick(gz2, 60, rng)]
rep["ids_same"] = l2 == [r["line"] for r in li]
rep["ids_strata"] = len(gz2)
rep["sha_now"] = {k: sha(PROC / n) for k, n in (("aihub_ref", "m32_stageA_aihub_ref.csv"), ("lines_4tu", "m32_stageA_lines_4tu.csv"), ("lines_ids", "m32_stageA_lines_ids.csv"))}
rep["sha_json"] = json.loads((PROC / "m32_texture_recipes.json").read_text(encoding="utf-8"))["list_sha256"]
rep["sha_match"] = rep["sha_now"] == rep["sha_json"]
rep["numpy"] = np.__version__
res["repro"] = rep

Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("done")
