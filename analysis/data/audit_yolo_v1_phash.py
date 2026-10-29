"""pHash near-duplicate audit across split boundaries (same gain only), full set."""
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
OUT = ROOT / "analysis" / "data" / "out"
m = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv")
m["part"] = np.where(m.subset == "test", "test", "fold" + m.fold.astype(str))

hs = np.zeros(len(m), dtype=np.uint64)
for i, n in enumerate(m.file_name):
    with Image.open(D / "images" / n) as im:
        h = imagehash.phash(im.convert("L"))
    bits = np.packbits(h.hash.flatten().astype(np.uint8))
    hs[i] = int.from_bytes(bits.tobytes(), "big")
m["phash"] = hs


def popcnt(x):
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


TH = [0, 2, 4, 8]
rows = []
examples = []
for g, gm in m.groupby("gain"):
    gm = gm.reset_index()
    h = gm.phash.values
    part = gm.part.values
    sess = gm.session.values
    pc = gm.primary_class.values
    n = len(gm)
    for i in range(n - 1):
        d = popcnt(h[i + 1:] ^ h[i])
        for t in TH[-1:]:
            js = np.nonzero(d <= t)[0]
            for j in js:
                jj = i + 1 + j
                rows.append((g, gm["index"][i], gm["index"][jj], int(d[j]), part[i], part[jj], pc[i], pc[jj], sess[i] == sess[jj]))
pairs = pd.DataFrame(rows, columns=["gain", "i", "j", "dist", "part_i", "part_j", "cls_i", "cls_j", "same_session"])
pairs["cross"] = pairs.part_i != pairs.part_j
pairs["cross_test_pool"] = (pairs.part_i == "test") ^ (pairs.part_j == "test")
pairs.to_csv(OUT / "phash_pairs_le8.csv", index=False)
print("images hashed", len(m))
print("pairs (dist<=8, same gain):", len(pairs))
for t in TH:
    p = pairs[pairs.dist <= t]
    print(f"dist<={t}: total {len(p)}, same_session {int(p.same_session.sum())}, cross-boundary {int(p.cross.sum())}, test-vs-pool {int(p.cross_test_pool.sum())}")
    c = p[p.cross]
    if len(c):
        c = c.assign(cls=np.where(c.cls_i == c.cls_j, c.cls_i, c.cls_i + "/" + c.cls_j))
        print("   cross by class:", c.cls.value_counts().to_dict(), "| cross & different session:", int((~c.same_session).sum()))
# summary of cross pairs dist<=4
c = pairs[(pairs.dist <= 4) & pairs.cross].sort_values("dist")
c["file_i"] = m.file_name.values[c.i.values]
c["file_j"] = m.file_name.values[c.j.values]
c.to_csv(OUT / "phash_cross_le4.csv", index=False, encoding="utf-8-sig")
print(c[["dist", "part_i", "part_j", "file_i", "file_j"]].head(15).to_string())
