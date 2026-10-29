"""Independent audit of data/morocco_eval_v1 (checklist M1-5, Morocco part).

Uses only pandas/csv/PIL/imagehash; does not import project code.
Writes analysis/data/out/morocco_eval_audit.json and a montage figure.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"C:\workspace\gpr_detection")
MOR = next(Path(r"C:\workspace\GPR_Data").glob("모로코*"))
EV = ROOT / "data" / "morocco_eval_v1"
OUT = ROOT / "analysis" / "data" / "out"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

idx = pd.read_csv(MOR / "01_메타데이터" / "images_index.csv", encoding="utf-8-sig")
grp = pd.read_csv(MOR / "01_메타데이터" / "split_groups.csv", encoding="utf-8-sig", dtype=str)
man = pd.read_csv(ROOT / "data" / "splits" / "morocco_eval_v1.csv", dtype={"stem": str})
res["index_rows"] = len(idx)
res["index_kinds"] = idx.kind.value_counts().to_dict()
yidx = idx[idx.yolo_path.notna()].copy()
res["yolo_images"] = len(yidx)
res["yolo_by_subset"] = yidx.subset.value_counts().to_dict()
res["aug_transform_counts"] = yidx.groupby(["subset", "aug_transform"]).size().rename("n").reset_index().values.tolist()

# --- 1. selection ---
rot_el = yidx.aug_transform.str.contains("회전|탄성")
res["rot_elastic_in_index"] = int(rot_el.sum())
res["rot_elastic_by_subset_aug"] = (
    yidx[rot_el].groupby(["subset", "aug_k"]).size().rename("n").reset_index().values.tolist()
)
# numbers whose transform is rotation/elastic, per subset
res["rot_elastic_numbers"] = {
    s: sorted(yidx[rot_el & (yidx.subset == s)].aug_k.unique().tolist()) for s in yidx.subset.unique()
}
res["manifest_rows"] = len(man)
res["manifest_unique_groups"] = int(man.group_id.nunique())
res["manifest_unique_files"] = int(man.file_name.nunique())
res["manifest_rot_elastic"] = int(man.aug_transform.str.contains("회전|탄성").sum())
sel = man.merge(yidx[["image_id", "aug_transform", "aug_k", "subset", "group_id", "stem", "split", "n_boxes", "width", "height"]],
                on="image_id", suffixes=("", "_idx"))
res["merge_rows"] = len(sel)
res["sel_aug_by_subset"] = man.groupby(["subset", "aug_number"]).size().rename("n").reset_index().values.tolist()
res["sel_transform_by_subset"] = man.groupby(["subset", "aug_transform"]).size().rename("n").reset_index().values.tolist()
res["idx_vs_manifest_mismatch"] = {
    "aug_transform": int((sel.aug_transform != sel.aug_transform_idx).sum()),
    "aug_number": int((sel.aug_number != sel.aug_k).sum()),
    "group": int((sel.group_id != sel.group_id_idx).sum()),
    "split": int((sel.split_src != sel.split).sum()),
    "size": int(((sel.width_px != sel.width) | (sel.height_px != sel.height)).sum()),
}
# independent selection: every group in split_groups has the 'almost identical' copy?
alm = yidx[yidx.aug_transform.str.contains("원본과 거의 동일")]
res["almost_identical_by_subset"] = alm.groupby("subset").size().to_dict()
res["almost_identical_aug_k"] = alm.groupby("subset").aug_k.unique().apply(lambda a: sorted(map(int, a))).to_dict()
res["split_groups_rows"] = len(grp)
res["split_groups_by_subset"] = grp.subset.value_counts().to_dict()
res["groups_missing_in_manifest"] = sorted(set(grp.group_id) - set(man.group_id))
res["groups_extra_in_manifest"] = sorted(set(man.group_id) - set(grp.group_id))
# groups in index vs split_groups
res["index_groups"] = int(yidx.group_id.nunique())
# stems per group (file order) and tie rule
g = grp.set_index("group_id")
multi = grp[grp.n_stems.astype(int) > 1]
res["multi_stem_groups"] = len(multi)
res["multi_stem_by_subset"] = multi.subset.value_counts().to_dict()
bad_first, details = [], []
m_stem = man.set_index("group_id").stem
for gid, row in multi.iterrows():
    pass
for _, row in multi.iterrows():
    stems = row.stems.replace(";", ",").replace("|", ",").split(",")
    stems = [s.strip() for s in stems]
    chosen = m_stem[row.group_id]
    details.append((row.group_id, row.stems, chosen))
    if chosen != stems[0]:
        bad_first.append(row.group_id)
res["multi_stem_sample"] = details[:6]
res["multi_stem_chosen_not_first"] = bad_first
res["group_stems_format_examples"] = multi.stems.head(5).tolist()
# are all copies of the other stems present (candidates available for alt stems)?
cand = {}
for _, row in multi.iterrows():
    stems = [s.strip() for s in row.stems.replace(";", ",").replace("|", ",").split(",")]
    pref = {"utilities": 6, "cavities": 13, "intact": 12}[row.subset]
    have = [s for s in stems if ((yidx.group_id == row.group_id) & (yidx.stem.astype(str) == s) & (yidx.aug_k == pref)).any()]
    cand[row.group_id] = have
res["multi_stem_groups_where_first_stem_lacks_preferred"] = [k for k, v in cand.items() if not v or v[0] != cand[k][0]]
res["multi_stem_candidate_counts"] = Counter(len(v) for v in cand.values())
# is the chosen stem also the lexicographically/numerically smallest?
def numkey(s):
    import re
    m = re.match(r"\d+", s)
    return (int(m.group()) if m else 10**9, s)
nonmin = []
for _, row in multi.iterrows():
    stems = [s.strip() for s in row.stems.replace(";", ",").replace("|", ",").split(",")]
    if man.set_index("group_id").stem[row.group_id] != min(stems, key=numkey):
        nonmin.append((row.group_id, row.stems))
res["multi_stem_chosen_not_numeric_min"] = nonmin
res["multi_stem_chosen_not_numeric_min_n"] = len(nonmin)

# --- 2. labels ---
sha_bad, lab_issues = [], []
cls_by_subset = {}
box_mismatch, deg, outr = [], 0, 0
nb_total = 0
sizes = Counter()
boxes_px = []
for _, r in man.iterrows():
    src = MOR / r.source_image
    dst = EV / "images" / r.file_name
    if hashlib.sha256(src.read_bytes()).hexdigest() != hashlib.sha256(dst.read_bytes()).hexdigest():
        sha_bad.append(r.file_name)
    with Image.open(dst) as im:
        sizes[(im.width, im.height, im.mode)] += 1
        W, H = im.size
    lab_src = MOR / "03_yolo_dataset" / "labels" / r.split_src / (r.image_id + ".txt")
    lab_dst = EV / "labels" / (r.image_id + ".txt")
    sl = [l.split() for l in lab_src.read_text(encoding="utf-8").splitlines() if l.strip()]
    dl = [l.split() for l in lab_dst.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(sl) != len(dl) or len(dl) != r.n_boxes:
        box_mismatch.append((r.file_name, len(sl), len(dl), r.n_boxes))
    expect = {"utilities": "0", "cavities": "2", "intact": None}[r.subset]
    for s, d in zip(sl, dl):
        nb_total += 1
        cls_by_subset.setdefault(r.subset, Counter())[d[0]] += 1
        if d[0] != expect:
            lab_issues.append((r.file_name, "class", d[0]))
        if [float(x) for x in s[1:]] != [float(x) for x in d[1:]]:
            lab_issues.append((r.file_name, "coords_differ"))
        v = [float(x) for x in d[1:]]
        if any(x < 0 or x > 1 for x in v):
            outr += 1
        xc, yc, w, h = v
        if w * W < 3 or h * H < 3:
            deg += 1
        if xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc - h / 2 < -1e-6 or yc + h / 2 > 1 + 1e-6:
            outr += 1
        boxes_px.append((r.subset, w * W, h * H))
    if r.subset == "intact" and (dl or lab_dst.stat().st_size != 0):
        lab_issues.append((r.file_name, "intact_nonempty"))
res["sha_mismatch"] = sha_bad
res["sizes"] = {str(k): v for k, v in sizes.items()}
res["label_box_count_mismatch"] = box_mismatch
res["label_issues"] = lab_issues
res["classes_by_subset"] = {k: dict(v) for k, v in cls_by_subset.items()}
res["boxes_total"] = nb_total
res["degenerate_lt3px"] = deg
res["out_of_range"] = outr
bp = pd.DataFrame(boxes_px, columns=["subset", "w", "h"])
res["box_px_stats"] = bp.groupby("subset").agg(w_min=("w", "min"), w_med=("w", "median"), h_min=("h", "min"), h_med=("h", "median")).round(1).to_dict("index")
res["files_in_dirs"] = {"images": len(list((EV / "images").glob("*.jpg"))), "labels": len(list((EV / "labels").glob("*.txt")))}
lst = (EV / "eval_images.txt").read_text(encoding="utf-8").splitlines()
res["eval_list_lines"] = len(lst)
res["eval_list_all_exist"] = all((EV / l).exists() for l in lst)
res["eval_yaml"] = (EV / "eval.yaml").read_text(encoding="utf-8")
res["dup_stem_in_group_files"] = int(man.file_name.duplicated().sum())

# --- 3. pHash ---
def ph(path):
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("L"))
    return int.from_bytes(np.packbits(h.hash.flatten().astype(np.uint8)).tobytes(), "big")

mh = np.array([ph(EV / "images" / n) for n in man.file_name], dtype=object)
n = len(mh)
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        d = bin(mh[i] ^ mh[j]).count("1")
        if d <= 8:
            pairs.append((man.file_name[i], man.file_name[j], man.subset[i], man.subset[j], man.split_src[i], man.split_src[j], d))
pd.DataFrame(pairs, columns=["a", "b", "sub_a", "sub_b", "split_a", "split_b", "dist"]).to_csv(OUT / "morocco_eval_phash_internal_le8.csv", index=False)
pa = pd.DataFrame(pairs, columns=["a", "b", "sub_a", "sub_b", "split_a", "split_b", "dist"])
res["internal_phash_pairs"] = {f"le{t}": int((pa.dist <= t).sum()) for t in (0, 2, 4, 6, 8)}
res["internal_phash_cross_split_le4"] = int(((pa.dist <= 4) & (pa.split_a != pa.split_b)).sum())

am = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv")
AD = ROOT / "data" / "aihub_yolo_v1" / "images"
ah = np.zeros(len(am), dtype=np.uint64)
for i, nme in enumerate(am.file_name):
    ah[i] = np.uint64(ph(AD / nme))
res["aihub_images_hashed"] = len(am)


def popcnt(x):
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


best = []
for i in range(n):
    d = popcnt(ah ^ np.uint64(mh[i]))
    k = int(d.argmin())
    best.append((man.file_name[i], man.subset[i], int(d[k]), am.file_name[k], am.subset[k] if "subset" in am else ""))
bd = pd.DataFrame(best, columns=["eval", "eval_subset", "min_dist", "aihub", "aihub_part"])
bd.to_csv(OUT / "morocco_eval_vs_aihub_nearest.csv", index=False)
res["vs_aihub_min_dist_overall"] = int(bd.min_dist.min())
res["vs_aihub_min_dist_hist"] = {f"le{t}": int((bd.min_dist <= t).sum()) for t in (0, 2, 4, 6, 8, 10, 12)}
res["vs_aihub_min_dist_median"] = float(bd.min_dist.median())
res["vs_aihub_closest5"] = bd.nsmallest(5, "min_dist").values.tolist()
res["vs_aihub_by_subset_min"] = bd.groupby("eval_subset").min_dist.agg(["min", "median"]).to_dict("index")
(OUT / "morocco_eval_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

# --- montage (local only; name contains 'montage') ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = ROOT / "docs" / "datasets" / "figures"
rng = np.random.default_rng(20261029)
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for r_i, sub in enumerate(["utilities", "cavities", "intact"]):
    s = man[man.subset == sub]
    pick = s.iloc[rng.choice(len(s), 6, replace=False)]
    for c_i, (_, r) in enumerate(pick.iterrows()):
        ax = axes[r_i, c_i]
        im = Image.open(EV / "images" / r.file_name).convert("L")
        ax.imshow(im, cmap="gray")
        W, H = im.size
        for l in (EV / "labels" / (r.image_id + ".txt")).read_text().splitlines():
            c, xc, yc, w, h = l.split()
            ax.add_patch(plt.Rectangle(((float(xc) - float(w) / 2) * W, (float(yc) - float(h) / 2) * H), float(w) * W, float(h) * H,
                                       fill=False, ec="red" if c == "0" else "cyan", lw=1.5))
        ax.set_title(f"{r.image_id}\n{r.split_src}, {r.n_boxes} box", fontsize=8)
        ax.axis("off")
fig.suptitle("morocco_eval_v1 samples (red=pipe_hyperbola, cyan=cavity; seed 20261029)")
fig.tight_layout()
fig.savefig(fig_dir / "morocco_eval_v1_samples_montage.png", dpi=110)
print("done")
