"""Cross-crop label consistency: consecutive 256-px crops on the same scan line overlap in x.
Compare pipe_band x-coverage inside the shared region (v1 subset only; neighbours missing from v1 are not seen)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
b = pd.read_csv(OUT / "pipe_band_boxes.csv")
man = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv", dtype={"fold": "string"})
rx = re.compile(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
ck = man.drop_duplicates("crop_key")[["crop_key", "session", "subset", "fold"]].copy()
m = ck.crop_key.str.extract(rx)
ck["line"], ck["a"], ck["b"] = m[0], m[1].astype(int), m[2].astype(int)
band = {k: g[["x1", "x2"]].values for k, g in b[b.cls == 1].groupby("crop_key")}
rows = []
strides = []
for ln, g in ck.groupby("line"):
    g = g.sort_values("a").reset_index(drop=True)
    for i in range(len(g) - 1):
        c1, c2 = g.loc[i], g.loc[i + 1]
        if c2.a >= c1.b:
            continue
        strides.append(c2.a - c1.a)
        lo, hi = c2.a, c1.b
        w = hi - lo

        def cov(c):
            iv = band.get(c.crop_key, np.zeros((0, 2)))
            mask = np.zeros(w, bool)
            for x1, x2 in iv:
                s, e = int(round(c.a + x1 - lo)), int(round(c.a + x2 - lo))
                mask[max(s, 0):max(min(e, w), 0)] = True
            return mask.mean(), mask
        f1, m1 = cov(c1)
        f2, m2 = cov(c2)
        rows.append((ln, c1.crop_key, c2.crop_key, c1.session, w, f1, f2, (m1 ^ m2).mean(), c1.crop_key in band, c2.crop_key in band))
d = pd.DataFrame(rows, columns=["line", "k1", "k2", "session", "overlap_w", "cov1", "cov2", "xor_frac", "has1", "has2"])
print("overlapping consecutive crop pairs", len(d), "stride quantiles", pd.Series(strides).quantile([0, .1, .5, .9, 1]).tolist())
print("overlap width quantiles", d.overlap_w.quantile([0, .1, .5, .9, 1]).tolist())
e = d[(d.has1 | d.has2) & (d.overlap_w >= 32)]
print("pairs with band label in at least one crop (overlap>=32):", len(e), " both have band:", int((e.has1 & e.has2).sum()),
      " only one:", int((e.has1 ^ e.has2).sum()))
bo = e[e.has1 & e.has2]
print("both labeled: xor_frac quantiles 50/75/90:", bo.xor_frac.quantile([.5, .75, .9]).round(3).tolist(), " pairs xor_frac>0.2:", int((bo.xor_frac > 0.2).sum()))
one = e[e.has1 ^ e.has2]
print("one-sided pairs: covered fraction in labeled side median", one[["cov1", "cov2"]].max(axis=1).median().round(3),
      " covered>=0.3 in the labeled side:", int((one[["cov1", "cov2"]].max(axis=1) >= 0.3).sum()))
print("by session (pairs with band in >=1 crop):")
print(e.groupby("session").agg(pairs=("k1", "size"), both=("has1", lambda s: 0)).join(
    bo.groupby("session").agg(both_labeled=("k1", "size"), xor_gt20=("xor_frac", lambda s: int((s > 0.2).sum())))).fillna(0).astype(int).to_string())
d.to_csv(OUT / "pipe_band_v2_crop_overlap.csv", index=False)
