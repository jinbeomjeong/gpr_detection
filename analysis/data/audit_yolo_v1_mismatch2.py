"""Reclassify filename mismatches (ignore '.jpg' suffix) and montage the real ones."""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
OUT = ROOT / "analysis" / "data" / "out"
FIG = ROOT / "docs" / "datasets" / "figures"
df = pd.read_csv(OUT / "filename_mismatch_in_dataset.csv")
df["xml_stem"] = df.xml_filename.str.replace(r"\.jpg$", "", regex=True)
df["real"] = df.xml_stem != df.member_stem
print("suffix-only:", int((~df.real).sum()), "real:", int(df.real.sum()))
r = df[df.real].copy()
print(r.groupby(["subset", "fold"], dropna=False).size().to_string())
print(r[["member_stem", "xml_stem", "pattern", "category", "subset", "fold"]].to_string())
print("by category (all mismatches):\n", df.groupby(["category", "real"]).size().to_string())
cols = ["#00ffff", "#ffff00", "#ff4040", "#40ff40"]
tiles = []
for _, x in r.iterrows():
    img = cv2.imread(str(D / "images" / (x.member_stem + ".jpg")))
    img = cv2.resize(img, (384, 384))
    cs = []
    for l in (D / "labels" / (x.member_stem + ".txt")).read_text().splitlines():
        c, a, b, w, h = l.split(); c = int(c); cs.append(c)
        a, b, w, h = [float(v) * 384 for v in (a, b, w, h)]
        col = tuple(int(cols[c][k:k + 2], 16) for k in (5, 3, 1))
        cv2.rectangle(img, (int(a - w / 2), int(b - h / 2)), (int(a + w / 2), int(b + h / 2)), col, 2)
    cv2.putText(img, f"cls{sorted(set(cs))} {x.pattern}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(img)
while len(tiles) % 5:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(FIG / "aihub_yolo_v1_filename_mismatch_real_montage.png"), np.vstack([np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)]))
print("saved", len(r))
