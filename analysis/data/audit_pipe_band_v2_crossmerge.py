"""Simulate cross-crop merging in absolute scan-line x coordinates (v1 subset only), then clip back to each crop.
Reports resulting box counts, widths, full-width fraction, chain length and union height inflation.
Pure statistics; no label file is written."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1].copy()
rx = re.compile(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
m = bb.crop_key.str.extract(rx)
bb["line"], bb["a"], bb["cb"] = m[0], m[1].astype(float), m[2].astype(float)
bb["ax1"] = bb.a + bb.x1
bb["ax2"] = bb.a + bb.x2
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def components(arr, G):
    n = len(arr)
    parent = list(range(n))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ih = min(arr[i, 3], arr[j, 3]) - max(arr[i, 1], arr[j, 1])
            if ih <= 0:
                continue
            if ih / min(arr[i, 3] - arr[i, 1], arr[j, 3] - arr[j, 1]) < 0.5:
                continue
            gap = max(arr[i, 0], arr[j, 0]) - min(arr[i, 2], arr[j, 2])
            if gap <= G:
                parent[f(i)] = f(j)
    return np.array([f(i) for i in range(n)])


for G in (0, 8, 16, 32):
    nb = 0
    widths = []
    chain = []
    infl = []
    crops_changed = 0
    per_crop_before = bb.groupby("crop_key").size().to_dict()
    after = {}
    for ln, g in bb.groupby("line"):
        arr = g[["ax1", "y1", "ax2", "y2"]].values
        if G == 0:
            comp = np.arange(len(arr))
        else:
            comp = components(arr, G)
        crops = g[["crop_key", "a", "cb"]].drop_duplicates()
        for c in np.unique(comp):
            idx = comp == c
            sub = arr[idx]
            x1, x2 = sub[:, 0].min(), sub[:, 2].max()
            y1, y2 = sub[:, 1].min(), sub[:, 3].max()
            chain.append(x2 - x1)
            infl.append((y2 - y1) / (sub[:, 3] - sub[:, 1]).max())
            for cr in crops.itertuples():
                lo, hi = max(x1, cr.a), min(x2, cr.cb)
                if hi - lo >= 8:
                    after.setdefault(cr.crop_key, []).append((lo - cr.a, hi - cr.a))
                    widths.append(hi - lo)
    # crops considered = crops with band boxes originally
    tot_after = sum(len(v) for v in after.values())
    full = np.mean(np.array(widths) >= 0.95 * 256)
    chg = sum(1 for k in per_crop_before if len(after.get(k, [])) != per_crop_before[k])
    P(f"G={G}: clipped boxes {tot_after} (before {sum(per_crop_before.values())}), crops with changed box count {chg}, "
      f"width p50/p90 {np.percentile(widths,50):.0f}/{np.percentile(widths,90):.0f}, full-width(>=95%) {full*100:.1f}%, "
      f"components {len(chain)}, chain len p50/p90/max {np.percentile(chain,50):.0f}/{np.percentile(chain,90):.0f}/{max(chain):.0f}, "
      f"height inflation p50/p95/max {np.percentile(infl,50):.2f}/{np.percentile(infl,95):.2f}/{max(infl):.2f}")
(OUT / "pipe_band_v2_crossmerge_report.txt").write_text("\n".join(lines), encoding="utf-8")
