"""Filename-mismatch records inside aihub_yolo_v1 + 20-sample montage."""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\workspace\gpr_detection\src")
import cv2
import numpy as np
import pandas as pd

from gpr_detection.io.aihub import AIHubSplit, AIHubZipReader, list_aihub_bscan_zip_pairs
from gpr_detection.io.paths import load_dataset_paths

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
OUT = ROOT / "analysis" / "data" / "out"
FIG = ROOT / "docs" / "datasets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv")
inset = set(m.file_name.str.replace(".jpg", "", regex=False))
root = load_dataset_paths().aihub

rows = []
total_mis = 0
total_sess = 0
for split in (AIHubSplit.TRAINING, AIHubSplit.VALIDATION):
    for pair in list_aihub_bscan_zip_pairs(root, split):
        with AIHubZipReader(pair) as r:
            for rec in r.iter_records():
                mis = rec.has_filename_mismatch
                total_mis += mis
                total_sess += rec.has_session_mismatch
                if mis and rec.member_stem in inset:
                    rows.append((rec.member_stem, rec.filename_field, rec.session, rec.session_from_name, rec.category, split.value))
df = pd.DataFrame(rows, columns=["member_stem", "xml_filename", "session", "session_from_name", "category", "split_src"])
df = df.merge(m[["file_name", "subset", "fold"]].assign(member_stem=m.file_name.str.replace(".jpg", "", regex=False)), on="member_stem", how="left")
print("total mismatch (all zips)", total_mis, "session mismatch (all)", total_sess)
print("mismatch in dataset", len(df))
print(df.groupby(["subset", "fold"], dropna=False).size().to_string())
print(df.category.value_counts().to_string())


def pattern(a, b):
    a = str(a); b = str(b)
    if a.lower() == b.lower():
        return "case"
    ga = re.sub(r"_g[0-9.]+_", "_", a); gb = re.sub(r"_g[0-9.]+_", "_", b)
    if ga == gb:
        return "gain_only"
    pa = re.sub(r"_x\(.*\)", "", ga); pb = re.sub(r"_x\(.*\)", "", gb)
    if pa == pb:
        return "x_range_differs"
    if re.sub(r"_B\d+", "", pa) == re.sub(r"_B\d+", "", pb):
        return "channel_differs"
    return "other"


df["pattern"] = [pattern(a, b) for a, b in zip(df.member_stem, df.xml_filename)]
print(df.pattern.value_counts().to_string())
df.to_csv(OUT / "filename_mismatch_in_dataset.csv", index=False, encoding="utf-8-sig")
print(df[["member_stem", "xml_filename"]].head(8).to_string())
fn_in = df.xml_filename.str.replace(".jpg", "", regex=False).isin(inset)
print("xml filename itself present in dataset:", int(fn_in.sum()))

rng = random.Random(20261029)
idx = list(range(len(df)))
rng.shuffle(idx)
sel = idx[:20]
cols = ["#00ffff", "#ffff00", "#ff4040", "#40ff40"]
tiles = []
for i in sel:
    r = df.iloc[i]
    img = cv2.imread(str(D / "images" / (r.member_stem + ".jpg")))
    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
    cs = []
    for l in (D / "labels" / (r.member_stem + ".txt")).read_text().splitlines():
        c, x, y, w, h = l.split(); c = int(c); cs.append(c)
        x, y, w, h = [float(v) * 384 for v in (x, y, w, h)]
        col = tuple(int(cols[c][k:k + 2], 16) for k in (5, 3, 1))
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), col, 2)
    cv2.putText(img, f"#{i} cls{sorted(set(cs))} {r.pattern}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(img)
    print(i, r.member_stem, "<- xml:", r.xml_filename, "|", r.pattern, "|", r.category)
rows_img = [np.hstack(tiles[k:k + 5]) for k in range(0, 20, 5)]
cv2.imwrite(str(FIG / "aihub_yolo_v1_filename_mismatch_montage.png"), np.vstack(rows_img))
print("montage saved")
