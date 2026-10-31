"""Sanity check: do overlapping crops share pixels at the stride offset, and how do their band boxes compare?"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
FIG = ROOT / "docs" / "datasets" / "figures" / "pipe_band_shape"
IMG = ROOT / "data" / "aihub_yolo_v1" / "images"
d = pd.read_csv(OUT / "pipe_band_v2_crop_overlap.csv")
b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1]
fo = b.drop_duplicates("crop_key").set_index("crop_key")["file"].to_dict()
bx = {k: g[["x1", "y1", "x2", "y2"]].values for k, g in bb.groupby("crop_key")}
import re
rx = re.compile(r"_x\((\d+)-(\d+)\)$")
cand = d[(d.xor_frac > 0.3) & (d.overlap_w >= 120) & d.has1 & d.has2].copy()
cand["sess_n"] = cand.groupby("session").cumcount()
pick = cand[cand.sess_n < 2].head(8)
diffs = []
fig, axs = plt.subplots(len(pick), 2, figsize=(10, 4.2 * len(pick) / 1.6), dpi=70)
for r, (i, row) in zip(range(len(pick)), pick.iterrows()):
    a1 = int(rx.search(row.k1).group(1))
    a2 = int(rx.search(row.k2).group(1))
    s = a2 - a1
    im1 = np.asarray(Image.open(IMG / fo[row.k1]).convert("L"), np.float32)
    im2 = np.asarray(Image.open(IMG / fo[row.k2]).convert("L"), np.float32)
    w = 256 - s
    # rows 0..255 compare overlapping region: im1[:, s:], im2[:, :w]
    md = np.abs(im1[:, s:] - im2[:, :w]).mean()
    diffs.append((row.k1, s, w, md))
    for ax, im, k, off in ((axs[r][0], im1, row.k1, 0), (axs[r][1], im2, row.k2, s)):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        for x1, y1, x2, y2 in bx.get(k, []):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ff3030", lw=1.2))
        ax.axvline(s if off == 0 else w, color="yellow", lw=0.6)
        ax.set_title(f"{k[-30:]} stride={s}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIG / "pipe_band_v2_crop_overlap_check.png")
for t in diffs:
    print(t)
