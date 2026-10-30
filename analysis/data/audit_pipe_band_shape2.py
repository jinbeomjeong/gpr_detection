"""Part 2: per-session fragmentation, merge simulation, depth, extreme lists, overlays.
Reads analysis/data/out/pipe_band_boxes.csv + pipe_band_pairs.csv (from audit_pipe_band_shape.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(r"C:\workspace\gpr_detection")
D = ROOT / "data" / "aihub_yolo_v1"
OUT = ROOT / "analysis" / "data" / "out"
FIG = ROOT / "docs" / "datasets" / "figures" / "pipe_band_shape"
FIG.mkdir(parents=True, exist_ok=True)
S = 256.0

b = pd.read_csv(OUT / "pipe_band_boxes.csv")
pr = pd.read_csv(OUT / "pipe_band_pairs.csv")
bb = b[b.cls == 1].copy()
lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)

# per-session fragmentation
bb["cy"] = (bb.y1 + bb.y2) / 2
imgs = bb.groupby(["session", "fold_lbl", "crop_key"]).size().rename("n").reset_index()
broken_imgs = set(pr[(pr.yov >= 0.5) & (pr.xgap > 0)].crop_key)
tight_imgs = set(pr[(pr.yov >= 0.5) & (pr.xgap > 0) & (pr.xgap <= 8)].crop_key)
imgs["broken"] = imgs.crop_key.isin(broken_imgs)
imgs["tight"] = imgs.crop_key.isin(tight_imgs)
sess = imgs.groupby(["session", "fold_lbl"]).agg(images=("n", "size"), multi=("n", lambda x: (x > 1).mean()),
                                                  broken=("broken", "mean"), tight=("tight", "mean"),
                                                  nbox_mean=("n", "mean")).round(3).reset_index()
sess = sess.sort_values("images", ascending=False)
P("== per-session fragmentation (broken = any pair y-overlap>=0.5 with x gap>0; tight = gap<=8px)")
P(sess.to_string(index=False))
P("overall: images", len(imgs), "broken", imgs.broken.sum(), f"({imgs.broken.mean():.3f})", "tight", imgs.tight.sum(), f"({imgs.tight.mean():.3f})")
P("session-level tight rate range:", sess.tight.min(), sess.tight.max())
P("boxes in tight/broken images:", bb.crop_key.isin(broken_imgs).sum(), "of", len(bb))

# gap histogram (y-overlap >=0.5)
g = pr[(pr.yov >= 0.5) & (pr.xgap > 0)]
P("gap quantiles", g.xgap.quantile([.1, .25, .5, .75, .9]).round(1).tolist())
P("gap<=2px:", (g.xgap <= 2).sum(), "<=4:", (g.xgap <= 4).sum(), "<=8:", (g.xgap <= 8).sum())

# merge simulation (greedy union-find over y-overlap>=0.5 & gap<=G)
def merge(df, G):
    outs = []
    for ck, gdf in df.groupby("crop_key"):
        r = gdf[["x1", "y1", "x2", "y2"]].values.tolist()
        changed = True
        while changed:
            changed = False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    a, c = r[i], r[j]
                    ih = min(a[3], c[3]) - max(a[1], c[1])
                    yov = ih / min(a[3] - a[1], c[3] - c[1])
                    xg = max(a[0], c[0]) - min(a[2], c[2])
                    if yov >= 0.5 and xg <= G:
                        r[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
                        r.pop(j)
                        changed = True
                        break
                if changed:
                    break
        for x in r:
            outs.append((ck, *x))
    return pd.DataFrame(outs, columns=["crop_key", "x1", "y1", "x2", "y2"])

P("\n== merge simulation")
P("baseline boxes", len(bb), "w med", bb.w.median(), "full-width frac", round(bb.full.mean(), 3))
for G in (8, 16, 32):
    mg = merge(bb, G)
    w = mg.x2 - mg.x1
    h = mg.y2 - mg.y1
    P(f"G={G}: boxes {len(mg)} (-{len(bb)-len(mg)}, {(len(bb)-len(mg))/len(bb):.3f}); w p25/50/75={w.quantile(.25):.0f}/{w.median():.0f}/{w.quantile(.75):.0f}; h med {h.median():.1f} p95 {h.quantile(.95):.1f}; full>=0.95: {(w/S>=0.95).mean():.3f}; images with 1 box: {(mg.groupby('crop_key').size()==1).mean():.3f}")

# depth
P("\n== depth (cy px) quantiles")
hy = b[b.cls == 0].copy()
hy["cy"] = (hy.y1 + hy.y2) / 2
P("band cy", bb.cy.quantile([.05, .25, .5, .75, .95]).round(0).tolist(), "hyperbola cy", hy.cy.quantile([.05, .25, .5, .75, .95]).round(0).tolist())

# stacked boxes in x-overlap & IoU>0
io = pr[pr.iou > 0].sort_values("iou", ascending=False)
P("\nIoU>0 pairs:", len(io), "max IoU", io.iou.max().round(3) if len(io) else None, "sessions", io.session.value_counts().to_dict())

# extremes lists
ext = []
cnt = bb.groupby("crop_key").size()
for ck in cnt.sort_values(ascending=False).head(10).index:
    ext.append(("most_boxes", ck, int(cnt[ck])))
tight_list = pr[(pr.yov >= 0.5) & (pr.xgap > 0)].groupby("crop_key").size().sort_values(ascending=False).head(10)
for ck, v in tight_list.items():
    ext.append(("most_broken_pairs", ck, int(v)))
for r in io.head(10).itertuples():
    ext.append(("iou_overlap", r.crop_key, round(r.iou, 3)))
nar = pd.read_csv(OUT / "pipe_band_confusion_band_narrow.csv")
for r in nar.sort_values("ar").head(10).itertuples():
    ext.append(("band_narrow_tall", r.crop_key, round(r.ar, 2)))
pd.DataFrame(ext, columns=["kind", "crop_key", "value"]).to_csv(OUT / "pipe_band_extremes.csv", index=False)
P("narrow-tall band session counts:", nar.session.value_counts().to_dict(), "fold", nar.fold_lbl.value_counts().to_dict())
P("narrow-tall band cy med", nar.y1.add(nar.y2).div(2).median())

# overlays
def find_file(ck):
    return b[b.crop_key == ck].file.iloc[0]

def overlay(ck, name):
    f = find_file(ck)
    im = Image.open(D / "images" / f).convert("RGB")
    dr = ImageDraw.Draw(im)
    for r in b[b.crop_key == ck].itertuples():
        col = (255, 60, 60) if r.cls == 1 else (60, 200, 255)
        dr.rectangle([r.x1, r.y1, r.x2, r.y2], outline=col, width=2)
    return im

# choose examples: one broken-tight, one many-boxes, one narrow-tall, one clean single, one hyperbola
rng = np.random.default_rng(20261029)
ex = []
tight_c = sorted(tight_imgs)
ex.append(("broken_tight", tight_c[int(rng.integers(len(tight_c)))]))
ex.append(("most_boxes", cnt.sort_values(ascending=False).index[0]))
ex.append(("band_narrow_tall", nar.sort_values("ar").crop_key.iloc[0]))
single = imgs[(imgs.n == 1)].crop_key.tolist()
ex.append(("single_clean", single[int(rng.integers(len(single)))]))
hyc = hy.crop_key.tolist()
ex.append(("hyperbola", hyc[int(rng.integers(len(hyc)))]))
if len(io):
    ex.append(("iou_overlap", io.crop_key.iloc[0]))
tiles = [overlay(ck, n) for n, ck in ex]
W = 256
sheet = Image.new("RGB", (W * 3, W * ((len(tiles) + 2) // 3)), (0, 0, 0))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * W, (i // 3) * W))
sheet.save(FIG / "pipe_band_examples.png")
P("\nexamples (order row-major):")
for n, ck in ex:
    P(" ", n, ck)
(OUT / "pipe_band_shape_report2.txt").write_text("\n".join(lines), encoding="utf-8")
