"""Shape/consistency audit of pipe_band (cls 1) vs pipe_hyperbola (cls 0) boxes in data/aihub_yolo_v1.

Single process, read-only. One representative image per crop_key (gain copies are collapsed).
Outputs: analysis/data/out/pipe_band_*.csv and pipe_band_shape_report.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
M = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
OUT = ROOT / "analysis" / "data" / "out"
OUT.mkdir(parents=True, exist_ok=True)
S = 256.0
EDGE = 1.5  # px tolerance for "touching the border"

m = pd.read_csv(M, dtype={"fold": "string"})
m["equip"] = m["file_name"].str.split("_").str[0]
# representative per crop_key: prefer gain 1.0
m["gpref"] = (m["gain"].astype(float) != 1.0).astype(int)
rep = m.sort_values(["crop_key", "gpref"]).drop_duplicates("crop_key")
print("manifest rows", len(m), "crop_keys", len(rep))

# check gain copies share identical labels (sample check over all crop_keys)
def read_label(fn):
    p = D / "labels" / (Path(fn).stem + ".txt")
    rows = []
    for ln in p.read_text().split():
        pass
    txt = p.read_text().strip().splitlines()
    for ln in txt:
        a = ln.split()
        rows.append((int(a[0]), *map(float, a[1:5])))
    return rows

diff_gain = 0
checked = 0
for ck, g in m.groupby("crop_key"):
    if len(g) < 2:
        continue
    sets = {tuple(sorted(read_label(f))) for f in g["file_name"]}
    checked += 1
    if len(sets) > 1:
        diff_gain += 1
print("crop_keys with >1 gain copy:", checked, "with differing labels:", diff_gain)

recs = []
for r in rep.itertuples():
    for cls, cx, cy, w, h in read_label(r.file_name):
        x1 = (cx - w / 2) * S
        x2 = (cx + w / 2) * S
        y1 = (cy - h / 2) * S
        y2 = (cy + h / 2) * S
        recs.append((r.crop_key, r.file_name, r.session, r.equip, r.subset, r.fold, cls,
                     x1, y1, x2, y2))
b = pd.DataFrame(recs, columns=["crop_key", "file", "session", "equip", "subset", "fold", "cls",
                                "x1", "y1", "x2", "y2"])
b["w"] = b.x2 - b.x1
b["h"] = b.y2 - b.y1
b["ar"] = b.w / b.h
b["area"] = b.w * b.h
b["wfrac"] = b.w / S
b["touch_x"] = (b.x1 <= EDGE) | (b.x2 >= S - EDGE)
b["touch_y"] = (b.y1 <= EDGE) | (b.y2 >= S - EDGE)
b["touch_l"] = b.x1 <= EDGE
b["touch_r"] = b.x2 >= S - EDGE
b["full"] = b.wfrac >= 0.95
b["fold_lbl"] = np.where(b.subset == "test", "test", "f" + b.fold.fillna("?"))
b.to_csv(OUT / "pipe_band_boxes.csv", index=False)

lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)

P("== crop images / boxes per class")
P(b.groupby("cls").agg(boxes=("w", "size"), images=("crop_key", "nunique"), sessions=("session", "nunique")).to_string())

P("\n== 1. shape distribution (px of 256)")
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
for c in (0, 1):
    x = b[b.cls == c]
    P(f"-- cls {c} n={len(x)}")
    for col in ["w", "h", "ar", "area", "wfrac"]:
        P(f"  {col}: " + " ".join(f"p{int(q*100)}={x[col].quantile(q):.2f}" for q in qs) + f" mean={x[col].mean():.2f}")
    P(f"  full-width(>=0.95): {x.full.mean():.3f}  touch_x: {x.touch_x.mean():.3f} (L {x.touch_l.mean():.3f}, R {x.touch_r.mean():.3f}) touch_y: {x.touch_y.mean():.3f}")
    P(f"  w bins frac: " + " ".join(f"{lo}-{hi}:{((x.wfrac>=lo)&(x.wfrac<hi)).mean():.3f}" for lo, hi in [(0,0.1),(0.1,0.25),(0.25,0.5),(0.5,0.75),(0.75,0.95),(0.95,1.01)]))

# 2. fragmentation
P("\n== 2. fragmentation (pipe_band)")
bb = b[b.cls == 1]
cnt = bb.groupby("crop_key").size()
P("boxes per image (band images):", cnt.value_counts().sort_index().to_dict())
hh = b[b.cls == 0].groupby("crop_key").size()
P("boxes per image (hyperbola images):", hh.value_counts().sort_index().head(12).to_dict())

pairs = []
for ck, g in bb.groupby("crop_key"):
    if len(g) < 2:
        continue
    rows = list(g.itertuples())
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, c = rows[i], rows[j]
            iw = min(a.x2, c.x2) - max(a.x1, c.x1)
            ih = min(a.y2, c.y2) - max(a.y1, c.y1)
            yov = ih / min(a.h, c.h)  # >0 when overlapping in y
            xgap = -iw  # >0: gap, <=0: overlap
            inter = max(iw, 0) * max(ih, 0)
            iou = inter / (a.area + c.area - inter)
            pairs.append((ck, a.file, a.session, a.equip, a.fold_lbl, xgap, yov, iou, ih, a.w, c.w))
pr = pd.DataFrame(pairs, columns=["crop_key", "file", "session", "equip", "fold_lbl", "xgap", "yov", "iou", "ih", "wa", "wb"])
pr.to_csv(OUT / "pipe_band_pairs.csv", index=False)
P("pairs total", len(pr))
if len(pr):
    sameband = pr[pr.yov >= 0.5]
    P("pairs with y-overlap>=50% of smaller height:", len(sameband))
    broken = sameband[sameband.xgap > 0]
    P("  of which x-gap>0 (broken in x):", len(broken), "gap<=8px:", (broken.xgap <= 8).sum(), "8-32px:", ((broken.xgap > 8) & (broken.xgap <= 32)).sum(), ">32px:", (broken.xgap > 32).sum())
    ov = sameband[sameband.xgap <= 0]
    P("  of which x-overlap (side by side stacked/overlapping):", len(ov))
    P("pairs IoU>0:", (pr.iou > 0).sum(), " IoU>0.3:", (pr.iou > 0.3).sum(), " IoU>0.7:", (pr.iou > 0.7).sum())
    P("images with >=1 IoU>0 pair:", pr[pr.iou > 0].crop_key.nunique(), " with >=1 broken-in-x pair:", broken.crop_key.nunique() if len(broken) else 0)
    P("images with >=1 y-overlap (>=0.5) pair, any x:", sameband.crop_key.nunique())
    P("pair ratio by fold (images with IoU>0 / band images):")
    for f, g in bb.groupby("fold_lbl"):
        P(f"   {f}: band images {g.crop_key.nunique()}, with IoU>0 pair {pr[(pr.fold_lbl==f)&(pr.iou>0)].crop_key.nunique()}, with broken-x pair {broken[broken.fold_lbl==f].crop_key.nunique()}")

# 3. per-session style
P("\n== 3. per-session style (band)")
def style(g):
    n = len(g)
    return pd.Series({
        "equip": g.equip.iloc[0], "fold": g.fold_lbl.iloc[0], "boxes": n, "images": g.crop_key.nunique(),
        "box_per_img": n / g.crop_key.nunique(), "w_med": g.w.median(), "h_med": g.h.median(),
        "full_frac": g.full.mean(), "touch_x": g.touch_x.mean(), "ar_med": g.ar.median(),
    })
ss = bb.groupby("session").apply(style, include_groups=False).reset_index()
def lab(r):
    if r.full_frac >= 0.8:
        return "full"
    if r.full_frac <= 0.2:
        return "short"
    return "mixed"
ss["style"] = ss.apply(lab, axis=1)
ss = ss.sort_values("boxes", ascending=False)
ss.to_csv(OUT / "pipe_band_session_style.csv", index=False)
P(ss.round(2).to_string(index=False))
P("style counts (sessions / boxes):")
P(ss.groupby("style").agg(sessions=("session", "size"), boxes=("boxes", "sum")).to_string())
P("equip counts band:", bb.groupby("equip").agg(boxes=("w", "size"), sessions=("session", "nunique")).to_dict())
P("equip counts hyperbola:", b[b.cls == 0].groupby("equip").agg(boxes=("w", "size"), sessions=("session", "nunique")).to_dict())
P("\n by-equip shape (band):")
P(bb.groupby("equip")[["w", "h", "ar"]].median().round(1).to_string())
P("\n by-equip shape (hyperbola):")
P(b[b.cls == 0].groupby("equip")[["w", "h", "ar"]].median().round(1).to_string())

P("\n== 3b. fold distribution")
fg = b[b.cls.isin([0, 1])].groupby(["fold_lbl", "cls"]).agg(
    boxes=("w", "size"), images=("crop_key", "nunique"), sessions=("session", "nunique"),
    w_med=("w", "median"), h_med=("h", "median"), full=("full", "mean"), touch=("touch_x", "mean")).round(2)
P(fg.to_string())
P("\nstyle per fold (sessions/boxes):")
P(pd.crosstab(ss.fold, ss.style).to_string())
P(ss.pivot_table(index="fold", columns="style", values="boxes", aggfunc="sum", fill_value=0).to_string())

# 4. confusion candidates
P("\n== 4. confusion candidates")
h = b[b.cls == 0]
P("hyperbola ar quantiles:", h.ar.quantile([0.05, 0.5, 0.95]).round(2).tolist(), "band ar:", bb.ar.quantile([0.05, 0.5, 0.95]).round(2).tolist())
# band boxes that look like hyperbolas: narrow (w<64) and ar<2.5
bn = bb[(bb.w < 64) & (bb.ar < 2.5)]
hw = h[(h.wfrac >= 0.5) | (h.ar >= 4)]
P("band narrow&tall (w<64px and ar<2.5):", len(bn), f"({len(bn)/len(bb):.3f}) images", bn.crop_key.nunique())
P("hyperbola wide (w>=128) or ar>=4:", len(hw), f"({len(hw)/len(h):.3f}) images", hw.crop_key.nunique())
P("hyperbola full-width (>=0.95):", h.full.sum())
# images with both classes
both = set(bb.crop_key) & set(h.crop_key)
P("images containing both band and hyperbola:", len(both))
# cross-class overlaps
xr = []
for ck in both:
    g = b[b.crop_key == ck]
    for a in g[g.cls == 1].itertuples():
        for c in g[g.cls == 0].itertuples():
            iw = min(a.x2, c.x2) - max(a.x1, c.x1)
            ih = min(a.y2, c.y2) - max(a.y1, c.y1)
            inter = max(iw, 0) * max(ih, 0)
            if inter > 0:
                xr.append((ck, inter / (a.area + c.area - inter)))
P("cross-class overlapping pairs (IoU>0):", len(xr), "IoU>0.5:", sum(1 for _, v in xr if v > 0.5))
bn.assign(kind="band_narrow").to_csv(OUT / "pipe_band_confusion_band_narrow.csv", index=False)
hw.assign(kind="hyp_wide").to_csv(OUT / "pipe_hyperbola_confusion_wide.csv", index=False)

# 5. extremes
P("\n== 5. extremes (band)")
tiny = bb[(bb.w < 16) | (bb.h < 4)]
P("tiny band boxes (w<16 or h<4):", len(tiny))
tall = bb.sort_values("h", ascending=False).head(8)
P("tallest:", tall[["file", "w", "h"]].round(0).to_string(index=False))
many = cnt.sort_values(ascending=False).head(8)
P("most boxes in image:", many.to_dict())
tiny.to_csv(OUT / "pipe_band_tiny.csv", index=False)
(OUT / "pipe_band_shape_report.txt").write_text("\n".join(lines), encoding="utf-8")
