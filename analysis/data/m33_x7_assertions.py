"""M3-3 X7 data re-assertions: Morocco eval set, 4TU duplicates / K canon, Guangzhou IDS list.

Ground truth content is NOT opened: label files are only hashed / sized; no depth table is read;
known_apexes.csv is only hashed and its row/line counts taken (coordinates are not printed or stored).
Single process, no randomness. Output: data/audit/m33_x7/x7_assertions.json (+ morocco_eval_sha.csv).
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(r"C:\workspace\gpr_detection")
GPR = Path(r"C:\workspace\GPR_Data")
MOR = GPR / "모로코_관로공동GPR영상_가공본_261016"
FOURTU = GPR / "4TU_네덜란드_매설관GPR_가공본_261015"
GZ = GPR / "광저우대_중국_관로GPR_가공본_261016"
OUT = REPO / "data" / "audit" / "m33_x7"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


res = {}
# ---------------------------------------------------------------- Morocco
man_p = REPO / "data/splits/morocco_eval_v1.csv"
man = pd.read_csv(man_p, encoding="utf-8", dtype={"stem": str})
idx_p = MOR / "01_메타데이터/images_index.csv"
grp_p = MOR / "01_메타데이터/split_groups.csv"
idx = pd.read_csv(idx_p, encoding="utf-8", dtype={"stem": str})
grp = pd.read_csv(grp_p, encoding="utf-8")
ev = REPO / "data/morocco_eval_v1"
rows, bad_img = [], 0
for r in man.itertuples():
    a = ev / "images" / r.file_name
    b = MOR / r.source_image
    sa, sb = sha(a), sha(b)
    lab = ev / "labels" / (Path(r.file_name).stem + ".txt")
    rows.append(dict(file_name=r.file_name, eval_sha256=sa, source_sha256=sb, equal=sa == sb,
                     label_sha256=sha(lab), label_bytes=lab.stat().st_size))
    bad_img += sa != sb
sh = pd.DataFrame(rows)
sh.to_csv(OUT / "morocco_eval_sha.csv", index=False, encoding="utf-8")
agg = lambda s: hashlib.sha256("\n".join(s).encode()).hexdigest()
m = {}
m["files_sha256"] = {"manifest": sha(man_p), "eval_images.txt": sha(ev / "eval_images.txt"), "eval.yaml": sha(ev / "eval.yaml"),
                     "summary.json": sha(ev / "summary.json"), "images_index.csv": sha(idx_p), "split_groups.csv": sha(grp_p),
                     "morocco_eval_sha.csv": sha(OUT / "morocco_eval_sha.csv")}
m["n_manifest"] = len(man)
m["n_eval_images_on_disk"] = len(list((ev / "images").glob("*.jpg")))
m["n_eval_labels_on_disk"] = len(list((ev / "labels").glob("*.txt")))
m["n_list_lines"] = len((ev / "eval_images.txt").read_text(encoding="utf-8").split())
m["image_sha_mismatch_vs_source"] = int(bad_img)
m["aggregate_sha256_of_image_shas_sorted_by_name"] = agg(sh.sort_values("file_name").eval_sha256)
m["aggregate_sha256_of_label_shas_sorted_by_name"] = agg(sh.sort_values("file_name").label_sha256)
m["empty_label_files_by_size"] = int((sh.label_bytes == 0).sum())
# forbidden copies
j = man.merge(idx[["image_id", "aug_transform", "aug_k", "subset", "group_id", "split"]], on="image_id", how="left", suffixes=("", "_idx"))
forb = lambda s: s.fillna("").str.contains("회전|탄성")
m["manifest_rows_not_in_index"] = int(j.aug_transform_idx.isna().sum())
m["manifest_forbidden_by_index_aug_transform"] = int(forb(j.aug_transform_idx).sum())
m["manifest_forbidden_by_manifest_column"] = int(forb(man.aug_transform).sum())
m["manifest_aug_transform_mismatch_vs_index"] = int((j.aug_transform != j.aug_transform_idx).sum())
m["manifest_group_mismatch_vs_index"] = int((j.group_id != j.group_id_idx).sum())
fi = idx[forb(idx.aug_transform)]
m["index_forbidden_total"] = len(fi)
m["index_forbidden_numbers_by_subset"] = {s: sorted(set(g.aug_k.astype(int))) for s, g in fi.groupby("subset")}
m["selected_aug_numbers_by_subset"] = {s: sorted(set(g.aug_number.astype(int))) for s, g in man.groupby("subset")}
m["selected_numbers_intersect_forbidden"] = int(sum(
    len(set(g.aug_number.astype(int)) & set(fi[fi.subset == s].aug_k.astype(int))) for s, g in man.groupby("subset")))
# one per group
m["split_groups_rows"] = len(grp)
m["manifest_unique_groups"] = int(man.group_id.nunique())
m["manifest_groups_with_not_exactly_one_image"] = int((man.group_id.value_counts() != 1).sum())
m["split_groups_not_in_manifest"] = int((~grp.group_id.isin(man.group_id)).sum())
m["manifest_groups_not_in_split_groups"] = int((~man.group_id.isin(grp.group_id)).sum())
m["subset_counts"] = man.subset.value_counts().to_dict()
m["anisotropic_stretch_true"] = int(man.anisotropic_stretch.sum())
m["anisotropic_by_subset"] = {s: int(g.anisotropic_stretch.sum()) for s, g in man.groupby("subset")}
m["split_src_counts"] = man.split_src.value_counts().to_dict()
# near duplicates
ph = {}
for r in man.itertuples():
    ph[r.image_id] = imagehash.phash(Image.open(ev / "images" / r.file_name))
ids = sorted(ph)
pairs = []
for i in range(len(ids)):
    for k in range(i + 1, len(ids)):
        d = int(ph[ids[i]] - ph[ids[k]])
        if d <= 8:
            pairs.append((ids[i], ids[k], d))
m["phash_pairs_le8"] = pairs
m["phash_pairs_le4"] = [p for p in pairs if p[2] <= 4]
u = man[man.image_id.isin(["U_233_aug_6", "U_239_aug_6"])][["image_id", "group_id", "split_src"]].to_dict("records")
m["U233_U239_rows"] = u
res["morocco"] = m

# ---------------------------------------------------------------- 4TU
f = {}
dup_p = REPO / "docs/datasets/fourtu_duplicate_radargrams.csv"
pre_p = REPO / "docs/processing/preregistration_4tu_reeval_v1.md"
lines_p = REPO / "data/processing/d3/lines.csv"
ka_p = REPO / "data/processing/d3/known_apexes.csv"
ridx_p = FOURTU / "01_메타데이터/radargrams_index.csv"
f["files_sha256"] = {"fourtu_duplicate_radargrams.csv": sha(dup_p), "preregistration_4tu_reeval_v1.md": sha(pre_p),
                     "d3/lines.csv": sha(lines_p), "d3/known_apexes.csv": sha(ka_p), "radargrams_index.csv(4TU)": sha(ridx_p)}
dup = pd.read_csv(dup_p, encoding="utf-8", dtype={"activity_a": str, "activity_b": str})
ridx = pd.read_csv(ridx_p, encoding="utf-8", dtype={"activity_id": str})
f["radargram_index_rows"] = len(ridx)
f["dup_pairs"] = len(dup)
f["dup_scope"] = dup.scope.value_counts().to_dict()
# hash npy for the whole set -> independent duplicate grouping
npy_sha = {}
for r in ridx.itertuples():
    p = FOURTU / "05_radargrams_npy" / r.activity_id / f"{r.path}.npy"
    npy_sha[(r.activity_id, r.path)] = sha(p)
grp_by = defaultdict(list)
for k, v in npy_sha.items():
    grp_by[v].append(k)
multi = {v: ks for v, ks in grp_by.items() if len(ks) > 1}
f["npy_files_hashed"] = len(npy_sha)
f["npy_byte_identical_groups"] = len(multi)
f["npy_byte_identical_group_sizes"] = dict(Counter(len(k) for k in multi.values()))
listed = {(r.activity_a, r.path_a, r.activity_b, r.path_b): r.sha256_npy for r in dup.itertuples()}
ok_sha = sum(npy_sha[(a, pa)] == s and npy_sha[(b, pb)] == s for (a, pa, b, pb), s in listed.items())
f["listed_pairs_sha_match_both_files"] = ok_sha
listed_files = {(a, pa) for a, pa, b, pb in listed} | {(b, pb) for a, pa, b, pb in listed}
unlisted = sum(1 for ks in multi.values() for k in ks if k not in listed_files)
f["byte_identical_files_not_in_list"] = unlisted
# within-activity copies: file_b is the copy
within = dup[dup.scope == "within_activity"]
copies = {(r.activity_b, r.path_b) for r in within.itertuples()}
f["within_copy_files"] = len(copies)
unique = ridx[[(a, p) not in copies for a, p in zip(ridx.activity_id, ridx.path)]]
f["lines_keep_both"] = len(unique)
# keep_one (independent re-implementation of prereg rule: strict superset wins, else lower activity number)
key = lambda a: tuple(int(x) for x in a.split("."))
cnt = unique.groupby("activity_id").size().to_dict()
shared = defaultdict(list)
for r in dup[dup.scope == "across_activities"].itertuples():
    lo, hi = sorted([(r.activity_a, r.path_a), (r.activity_b, r.path_b)], key=lambda t: key(t[0]))
    shared[(lo[0], hi[0])].append((lo[1], hi[1]))
dropped = set()
removed_acts = []
dec = []
for (lo, hi), ps in sorted(shared.items(), key=lambda kv: key(kv[0][0])):
    n = len(ps)
    nl, nh = cnt[lo], cnt[hi]
    if n == nl < nh:
        kept, drop, why, side = hi, lo, "strict_superset", 0
    elif n == nh < nl:
        kept, drop, why, side = lo, hi, "strict_superset", 1
    elif n == nl == nh:
        kept, drop, why, side = lo, hi, "equal_sets", 1
    else:
        kept, drop, why, side = lo, hi, "lower_number", 1
    for p in ps:
        dropped.add((drop, p[side]))
    if cnt[drop] == n:
        removed_acts.append(drop)
    dec.append(dict(activities=[lo, hi], n_shared=n, n_lines=[nl, nh], kept=kept, dropped=drop, reason=why))
f["keep_one_decisions"] = len(dec)
f["keep_one_dropped_lines"] = len(dropped)
f["lines_keep_one"] = len(unique) - len(dropped)
f["keep_one_activities_removed"] = sorted(removed_acts, key=key)
f["keep_one_reason_counts"] = dict(Counter(d["reason"] for d in dec))
pd.DataFrame([dict(activity_id=a, path=p) for a, p in sorted(dropped, key=lambda t: (key(t[0]), t[1]))]).to_csv(
    OUT / "fourtu_keep_one_dropped_lines.csv", index=False, encoding="utf-8")
f["keep_one_dropped_lines_csv_sha256"] = sha(OUT / "fourtu_keep_one_dropped_lines.csv")
# K canon: counts only
ka = pd.read_csv(ka_p, encoding="utf-8")
ln = pd.read_csv(lines_p, encoding="utf-8", dtype={"activity_id": str})
f["known_apexes_rows"] = len(ka)
f["known_apexes_unique_lines"] = int(ka.line.nunique())
f["d3_lines_rows"] = len(ln)
f["d3_lines_bundles_value_counts"] = ln.bundles.value_counts().to_dict()
res["fourtu"] = f

# ---------------------------------------------------------------- Guangzhou IDS
g = {}
st_p = REPO / "analysis/processing/m32_stageA_lines_ids.csv"
gi_p = GZ / "01_메타데이터/radargrams_index.csv"
g["files_sha256"] = {"m32_stageA_lines_ids.csv": sha(st_p), "radargrams_index.csv(광저우대)": sha(gi_p)}
st = pd.read_csv(st_p, encoding="utf-8")
gi = pd.read_csv(gi_p, encoding="utf-8")
g["stageA_rows"] = len(st)
g["stageA_unique"] = int(st.line.nunique())
g["stageA_in_index"] = int(st.line.isin(gi.file_id).sum())
sel = gi[gi.file_id.isin(st.line)]
g["stageA_format_counts"] = sel.format.value_counts().to_dict()
g["stageA_target_counts"] = sel.target.value_counts().to_dict()
g["stageA_antenna_counts"] = {str(k): int(v) for k, v in sel.antenna_MHz.value_counts().items()}
g["index_rows"] = len(gi)
g["index_IDS_rows"] = int((gi.format == "IDS .dt").sum())
res["guangzhou"] = g

(OUT / "x7_assertions.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
