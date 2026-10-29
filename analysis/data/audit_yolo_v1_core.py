"""Core audit of data/aihub_yolo_v1: summary vs manifest, leakage, pairs, labels, yaml."""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
M = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
OUT = ROOT / "analysis" / "data" / "out"
OUT.mkdir(parents=True, exist_ok=True)

s = json.loads((D / "summary.json").read_text(encoding="utf-8"))
print("summary keys:", list(s.keys()))
for k in s:
    if k not in ("folds", "balance", "source", "class_names"):
        v = s[k]
        print(k, json.dumps(v, ensure_ascii=False)[:600])

m = pd.read_csv(M, dtype={"fold": "Int64"})
print("manifest rows", len(m), "columns", list(m.columns))
print(m["subset"].value_counts().to_string())
print("seed unique", m["seed"].unique(), "attempt", m["test_attempt"].unique())
print("fold counts (trainpool):")
print(m[m.subset == "trainpool"]["fold"].value_counts().sort_index().to_string())
print("test fold values:", m[m.subset == "test"]["fold"].unique())

# class share in test
def has(c):
    return m["classes"].astype(str)
tp = m[m.subset == "test"]
print("test primary_class:\n", tp["primary_class"].value_counts().to_string())
print("trainpool primary_class:\n", m[m.subset == "trainpool"]["primary_class"].value_counts().to_string())
# pipe share of tpipe/lpipe sessions in test vs all
for cat in ["tpipe", "lpipe"]:
    pass
print("category x subset:")
print(pd.crosstab(m["category"], m["subset"]).to_string())

# leakage
def inter(a, b, col):
    return len(set(a[col]) & set(b[col]))

test = m[m.subset == "test"]
pool = m[m.subset == "trainpool"]
res = {}
for col in ["session", "session_from_name", "crop_key", "split_group"]:
    res[f"test_vs_pool/{col}"] = inter(test, pool, col)
    for i in range(5):
        for j in range(i + 1, 5):
            res[f"fold{i}_vs_fold{j}/{col}"] = inter(pool[pool.fold == i], pool[pool.fold == j], col)
    for i in range(5):
        res[f"test_vs_fold{i}/{col}"] = inter(test, pool[pool.fold == i], col)
bad = {k: v for k, v in res.items() if v}
print("leak pairs checked:", len(res), "nonzero:", bad)
# also cross: session of one vs session_from_name of other
for a_name, a in [("test", test)] + [(f"fold{i}", pool[pool.fold == i]) for i in range(5)]:
    for b_name, b in [("test", test)] + [(f"fold{i}", pool[pool.fold == i]) for i in range(5)]:
        if a_name >= b_name:
            continue
        x = len(set(a.session) & set(b.session_from_name)) + len(set(a.session_from_name) & set(b.session))
        if x:
            print("CROSS leak", a_name, b_name, x)
# stem-level crop_key consistent with gain removal
import re
ck = m["file_name"].str.replace(r"_g[0-9.]+_x", "_x", regex=True).str.replace(".jpg", "", regex=False)
print("crop_key mismatch with file_name-derived:", int((ck != m["crop_key"]).sum()))
print("duplicate file_name:", int(m["file_name"].duplicated().sum()))
# crop gain completeness
g = m.groupby("crop_key")["gain"].nunique()
print("crops with gain count distribution:", g.value_counts().to_dict())
# crop spans multiple fold/subset
sp = m.assign(part=m["subset"] + m["fold"].astype(str)).groupby("crop_key")["part"].nunique()
print("crops spanning >1 part:", int((sp > 1).sum()))

# pairs & labels
imgs = sorted(p.name for p in (D / "images").iterdir())
lbls = sorted(p.name for p in (D / "labels").iterdir())
print("images", len(imgs), "labels", len(lbls))
istem = {Path(n).stem for n in imgs}
lstem = {Path(n).stem for n in lbls}
print("img w/o label", len(istem - lstem), "label w/o img", len(lstem - istem))
print("manifest vs images dir diff:", len(set(m.file_name) ^ set(imgs)))

nbox = 0
degen = 0
cls_count = {}
oor = 0
badfmt = 0
empty = 0
size_bad = 0
sizes = {}
mix = []
for n in imgs:
    st = Path(n).stem
    lines = [l for l in (D / "labels" / (st + ".txt")).read_text().splitlines() if l.strip()]
    if not lines:
        empty += 1
    cs = set()
    with Image.open(D / "images" / n) as im:
        W, H = im.size
    sizes[(W, H)] = sizes.get((W, H), 0) + 1
    for l in lines:
        p = l.split()
        if len(p) != 5:
            badfmt += 1
            continue
        c = int(p[0]); x, y, w, h = map(float, p[1:])
        cs.add(c)
        cls_count[c] = cls_count.get(c, 0) + 1
        nbox += 1
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x - w / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6 and y - h / 2 >= -1e-6 and y + h / 2 <= 1 + 1e-6):
            oor += 1
        if w * W < 3 or h * H < 3:
            degen += 1
    if len(cs) > 1:
        mix.append((n, sorted(cs)))
print("boxes", nbox, "class counts", dict(sorted(cls_count.items())), "empty labels", empty, "badfmt", badfmt)
print("out-of-range boxes", oor, "degenerate(<3px)", degen)
print("image sizes", sizes)
print("mixed-class images", len(mix))
pd.DataFrame(mix, columns=["file_name", "classes"]).to_csv(OUT / "mixed_class_images.csv", index=False, encoding="utf-8-sig")
for n, c in mix:
    print("  ", n, c)

# filename mismatch inclusion: needs raw XML -> done in separate script
# yaml via ultralytics
from ultralytics.data.utils import check_det_dataset
for name in [f"fold{i}.yaml" for i in range(5)] + ["test.yaml"]:
    try:
        d = check_det_dataset(str(D / name), autodownload=False)
        def cnt(x):
            if isinstance(x, (list, tuple)):
                return sum(cnt(i) for i in x)
            p = Path(x)
            if p.suffix == ".txt":
                return len([l for l in p.read_text().splitlines() if l.strip()])
            return len(list(p.glob("*")))
        print(name, "train", cnt(d.get("train")) if d.get("train") else None, "val", cnt(d.get("val")) if d.get("val") else None, "names", d["names"], "nc", d["nc"])
    except Exception as e:
        print(name, "ERROR", repr(e)[:300])
# fold txt line counts vs manifest
for i in range(5):
    n = len([l for l in (D / f"fold{i}_images.txt").read_text().splitlines() if l.strip()])
    print("fold", i, "txt lines", n, "manifest", int(((m.subset == "trainpool") & (m.fold == i)).sum()))
print("test txt", len([l for l in (D / "test_images.txt").read_text().splitlines() if l.strip()]))
