"""Draft Label Studio import files for the pipe_band relabel (NOT imported anywhere; local-only output).

One task per crop_key (representative = gain 1.0 file when present). Pre-annotation = v1 boxes of the other classes
(unchanged, for context) + pipe_band boxes auto-merged (same row y-overlap >= 0.5, x gap <= G, iterative).
Waves: W0 pilot (stratified, seed 20261031), W1 high-fragmentation sessions, W2 rest of train_pool, W3 test.
Output: labeling/pipe_band_relabel/local/tasks_all.json, tasks_pilot.json, tasks_index.csv  (git-ignored: box coords of AIHub labels)
"""
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
LOC = ROOT / "labeling" / "pipe_band_relabel" / "local"
LOC.mkdir(parents=True, exist_ok=True)
G = 32
SEED = 20261031
S = 256.0
NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}


def merge_boxes(rows, G):
    boxes = [list(r) for r in rows]
    changed = True
    while changed:
        changed = False
        order = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                ih = min(a[3], c[3]) - max(a[1], c[1])
                if ih <= 0:
                    continue
                if ih / min(a[3] - a[1], c[3] - c[1]) >= 0.5:
                    gap = max(a[0], c[0]) - min(a[2], c[2])
                    if gap <= G:
                        order.append((gap, i, j))
        if order:
            _, i, j = sorted(order)[0]
            a, c = boxes[i], boxes[j]
            boxes[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
            boxes.pop(j)
            changed = True
    return boxes


def rect(box, label, i):
    x1, y1, x2, y2 = box
    return {"id": f"r{i}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": 256, "original_height": 256, "image_rotation": 0,
            "value": {"x": x1 / S * 100, "y": y1 / S * 100, "width": (x2 - x1) / S * 100, "height": (y2 - y1) / S * 100,
                      "rotation": 0, "rectanglelabels": [label]}}


b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1]
tiers = pd.read_csv(OUT / "pipe_band_v2_crop_tiers.csv").set_index("crop_key")
rep_file = bb.drop_duplicates("crop_key").set_index("crop_key")["file"].to_dict()
sess = bb.drop_duplicates("crop_key").set_index("crop_key")["session"].to_dict()
fold = bb.drop_duplicates("crop_key").set_index("crop_key")["fold_lbl"].to_dict()
cbox = {k: g[["x1", "y1", "x2", "y2"]].values.tolist() for k, g in bb.groupby("crop_key")}
DIRL = ROOT / "data" / "aihub_yolo_v1" / "labels"

HIGH = {"NM_2022-09-22_(11_10_46)", "NM_2022-10-19_(12_33_02)", "NM_2022-11-02_(14_16_20)"}
rows = []
for k in cbox:
    wave = "W3" if fold[k] == "test" else ("W1" if sess[k] in HIGH else "W2")
    rows.append((k, rep_file[k], sess[k], fold[k], tiers.loc[k, "tier"], int(tiers.loc[k, "files"]), wave))
idx = pd.DataFrame(rows, columns=["crop_key", "file", "session", "fold", "tier", "n_files", "wave"])

# pilot: 40 crops, stratified over (fold group x tier), test excluded from pilot (rule is frozen on train_pool first)
rng = np.random.default_rng(SEED)
pool = idx[idx.wave != "W3"].copy()
quota = {"A_gapped": 20, "B_multi_nogap": 6, "C_single": 14}
pilot = []
for t, q in quota.items():
    g = pool[pool.tier == t]
    # round-robin over sessions
    per = {s: x.sample(frac=1, random_state=SEED).crop_key.tolist() for s, x in g.groupby("session")}
    n = 0
    while n < q and any(per.values()):
        for s in list(per):
            if per[s] and n < q:
                pilot.append(per[s].pop())
                n += 1
idx["pilot"] = idx.crop_key.isin(pilot)
idx.loc[idx.pilot, "wave"] = "W0"
order = {"W0": 0, "W1": 1, "W2": 2, "W3": 3}
idx["w"] = idx.wave.map(order)
idx = idx.sort_values(["w", "session", "crop_key"]).reset_index(drop=True)


def make_task(r):
    k = r.crop_key
    other = []
    lab = DIRL / (Path(r.file).stem + ".txt")
    for ln in lab.read_text().strip().splitlines():
        a = ln.split()
        c = int(a[0])
        if c == 1:
            continue
        cx, cy, w, h = map(float, a[1:5])
        other.append([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S, c])
    merged = merge_boxes(cbox[k], G)
    res = [rect(o[:4], NAMES[o[4]], i) for i, o in enumerate(other)]
    res += [rect(m, "pipe_band", 100 + i) for i, m in enumerate(merged)]
    return {"data": {"image": "/data/local-files/?d=aihub_yolo_v1/images/" + quote(r.file, safe=""),
                     "crop_key": k, "file": r.file, "session": r.session, "fold": r.fold, "tier": r.tier,
                     "wave": r.wave, "n_files": r.n_files, "v1_band_boxes": len(cbox[k]), "pre_band_boxes": len(merged)},
            "predictions": [{"model_version": f"v1_merge_g{G}", "score": 1.0, "result": res}]}


tasks = [make_task(r) for r in idx.itertuples()]
(LOC / "tasks_all.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
pt = [t for t in tasks if t["data"]["wave"] == "W0"]
(LOC / "tasks_pilot.json").write_text(json.dumps(pt, ensure_ascii=False), encoding="utf-8")
idx.drop(columns=["w"]).to_csv(LOC / "tasks_index.csv", index=False)
print("tasks", len(tasks), "pilot", len(pt))
print(idx.groupby(["wave", "tier"]).size().unstack(fill_value=0))
print(idx.groupby("wave").agg(crops=("crop_key", "size"), files=("n_files", "sum")))
print("pre-annotation boxes", sum(t["data"]["pre_band_boxes"] for t in tasks), "v1", sum(t["data"]["v1_band_boxes"] for t in tasks))
