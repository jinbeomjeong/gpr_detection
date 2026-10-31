"""pipe_band label audit v2 - statistics (CPU only, read-only on data/aihub_yolo_v1 derived files).

Inputs : analysis/data/out/pipe_band_boxes.csv (from audit_pipe_band_shape.py, one crop per crop_key),
         data/splits/aihub_yolo_v1.csv, analysis/detection/m17_pipe_band_fp_list.csv
Outputs: analysis/data/out/pipe_band_v2_*.csv, pipe_band_v2_report.txt (local only, box coordinates)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
S = 256.0
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1].copy()
man = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv", dtype={"fold": "string"})
fp = pd.read_csv(ROOT / "analysis" / "detection" / "m17_pipe_band_fp_list.csv")
fp["crop_key"] = fp["image"].map(lambda p: Path(p).stem.replace("_g1.0", "").replace("_g0.5", "").replace("_g0.25", ""))
fp["gain"] = fp["image"].map(lambda p: re.search(r"_g([0-9.]+)_x", p).group(1))
P("FP rows", len(fp), "gain copies used:", fp.gain.value_counts().to_dict())


def merge_boxes(rows, G):
    """rows: list of (x1,y1,x2,y2). Merge same-depth (y overlap >= 0.5 of smaller height) boxes whose x gap <= G; iterate."""
    boxes = [list(r) for r in rows]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                ih = min(a[3], c[3]) - max(a[1], c[1])
                hmin = min(a[3] - a[1], c[3] - c[1])
                gap = max(a[0], c[0]) - min(a[2], c[2])
                if ih / hmin >= 0.5 and gap <= G:
                    boxes[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def iou(a, c):
    iw = min(a[2], c[2]) - max(a[0], c[0])
    ih = min(a[3], c[3]) - max(a[1], c[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter)


# ---- per-crop pair table with coordinates -------------------------------------------------
pairs = []
for ck, g in bb.groupby("crop_key"):
    rows = list(g.itertuples())
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, c = rows[i], rows[j]
            ih = min(a.y2, c.y2) - max(a.y1, c.y1)
            yov = ih / min(a.h, c.h)
            gap = max(a.x1, c.x1) - min(a.x2, c.x2)
            pairs.append((ck, a.session, a.fold_lbl, gap, yov, a.x1, a.y1, a.x2, a.y2, c.x1, c.y1, c.x2, c.y2))
pr = pd.DataFrame(pairs, columns=["crop_key", "session", "fold_lbl", "xgap", "yov", "ax1", "ay1", "ax2", "ay2", "bx1", "by1", "bx2", "by2"])
same = pr[(pr.yov >= 0.5) & (pr.xgap > 0)]
frag8 = set(same[same.xgap <= 8].crop_key)
frag_any = set(same.crop_key)
frag_gt8 = set(same[same.xgap > 8].crop_key)
multi = set(bb.groupby("crop_key").size().loc[lambda s: s >= 2].index)
P("band crops", bb.crop_key.nunique(), "multi-box", len(multi), "same-depth gapped pair crops", len(frag_any),
  "gap<=8", len(frag8), "gap>8", len(frag_gt8), "gap>8 only (no gap<=8)", len(frag_gt8 - frag8))
pr[(pr.yov >= 0.5) & (pr.xgap > 8)].to_csv(OUT / "pipe_band_v2_gap_gt8_pairs.csv", index=False)

# ---- verify merge simulation vs v1 ---------------------------------------------------------
crop_boxes = {ck: g[["x1", "y1", "x2", "y2"]].values.tolist() for ck, g in bb.groupby("crop_key")}
merged = {}
for G in (0, 8, 16, 32):
    tot = 0
    for ck, rows in crop_boxes.items():
        mb = merge_boxes(rows, G) if G else rows
        merged[(G, ck)] = mb
        tot += len(mb)
    P(f"G={G}: boxes {tot}")

# ---- link M1-7 FP list to v1 fragmentation --------------------------------------------------
sess_of = bb.drop_duplicates("crop_key").set_index("crop_key")["session"].to_dict()
fold_of = bb.drop_duplicates("crop_key").set_index("crop_key")["fold_lbl"].to_dict()
fp["in_band_crop"] = fp.crop_key.isin(crop_boxes)
fp["multi"] = fp.crop_key.isin(multi)
fp["frag_any"] = fp.crop_key.isin(frag_any)
fp["frag8"] = fp.crop_key.isin(frag8)
fp["frag_gt8_only"] = fp.crop_key.isin(frag_gt8 - frag8)
fp["session"] = fp.crop_key.map(sess_of)
fp["fold_lbl"] = fp.crop_key.map(fold_of)
for G in (8, 16, 32):
    best = []
    for r in fp.itertuples():
        mb = merged.get((G, r.crop_key), [])
        best.append(max([iou([r.x1, r.y1, r.x2, r.y2], m) for m in mb], default=0.0))
    fp[f"iou_merged{G}"] = best
fp.drop(columns=["image"]).to_csv(OUT / "pipe_band_v2_fp_linked.csv", index=False)
P("\n== FP category x v1 fragmentation (rows=category)")
cat = fp.groupby("category").agg(n=("crop_key", "size"), crops=("crop_key", "nunique"), in_band_crop=("in_band_crop", "sum"),
                                 multi=("multi", "sum"), frag_any=("frag_any", "sum"), frag8=("frag8", "sum"),
                                 frag_gt8_only=("frag_gt8_only", "sum"))
P(cat.to_string())
P("\nFP -> IoU>=0.45 with a merged GT box (would become matchable / duplicate under merge)")
for G in (8, 16, 32):
    P(f" G={G}:", fp.groupby("category").apply(lambda g: int((g[f"iou_merged{G}"] >= 0.45).sum()), include_groups=False).to_dict())
fragfp = fp[fp.category == "fragment"]
P("fragment FP crops:", fragfp.crop_key.nunique(), "of which crop has frag8:", fragfp[fragfp.frag8].crop_key.nunique(),
  "has any gapped same-depth pair:", fragfp[fragfp.frag_any].crop_key.nunique(),
  "multi-box but no gapped pair:", fragfp[(fragfp.multi) & (~fragfp.frag_any)].crop_key.nunique(),
  "single-box crop:", fragfp[~fragfp.multi].crop_key.nunique())
P("fragment FP by session:", fragfp.session.value_counts().to_dict())
# fold 0 crops fraction
f0 = bb[bb.fold_lbl == "f0"]
f0_crops = f0.crop_key.nunique()
P("fold0 band crops", f0_crops, "frag8 crops", len(frag8 & set(f0.crop_key)), "frag_any", len(frag_any & set(f0.crop_key)))
base = f0.crop_key.nunique()
fragcrops_f0 = set(f0.crop_key) & frag_any
P("fold0: share of fragment-FP crops among fragmented band crops:", fragfp.crop_key.nunique(), "/", len(fragcrops_f0))

# FP count per crop in fold0 session NM_20220922(111046) vs band crop count
s0 = "NM_2022-09-22_(11_10_46)"
P("session", s0, "FPs:", int((fp.session == s0).sum()), "band crops in fold0:", int((f0.session == s0).sum() and f0[f0.session == s0].crop_key.nunique()))

# FP "no_gt" same-row analysis
ng = fp[fp.category == "no_gt"].copy()
P("\n== no_gt FP:", len(ng), "in band-GT crops:", int(ng.in_band_crop.sum()))
rows = []
for r in ng.itertuples():
    gts = crop_boxes.get(r.crop_key, [])
    best = None
    for gx in gts:
        ih = min(r.y2, gx[3]) - max(r.y1, gx[1])
        yov = ih / min(r.y2 - r.y1, gx[3] - gx[1])
        gap = max(r.x1, gx[0]) - min(r.x2, gx[2])
        if yov > 0 and (best is None or gap < best[0]):
            best = (gap, yov, gx)
    rows.append((r.crop_key, r.session, r.x1, r.y1, r.x2, r.y2, r.conf if hasattr(r, "conf") else r.conf,
                 best[0] if best else np.nan, best[1] if best else np.nan, *(best[2] if best else [np.nan] * 4)))
ngt = pd.DataFrame(rows, columns=["crop_key", "session", "px1", "py1", "px2", "py2", "conf", "gap_to_gt", "yov", "gx1", "gy1", "gx2", "gy2"])
ngt["same_row"] = ngt.yov.fillna(0) > 0
ngt.to_csv(OUT / "pipe_band_v2_nogt_fp.csv", index=False)
P("same-row (y overlap>0 with a GT band box):", int(ngt.same_row.sum()))
sr = ngt[ngt.same_row]
P("  gap_to_gt quantiles (px) 10/25/50/75/90:", sr.gap_to_gt.quantile([.1, .25, .5, .75, .9]).round(1).tolist())
P("  gap<=8:", int((sr.gap_to_gt <= 8).sum()), " 8-32:", int(((sr.gap_to_gt > 8) & (sr.gap_to_gt <= 32)).sum()), " >32:", int((sr.gap_to_gt > 32).sum()))
P("  by session:", sr.session.value_counts().to_dict())

# ---- per-session x fold counts --------------------------------------------------------------
P("\n== per-session pipe_band stats by fold")
fcnt = man.groupby("crop_key").size()
man_band = man[man.classes.astype(str).str.contains("1")]  # not reliable; use boxes instead
sess_rows = []
for (fl, ss), g in bb.groupby(["fold_lbl", "session"]):
    crops = set(g.crop_key)
    n_files = int(man[man.crop_key.isin(crops)].shape[0])
    sess_rows.append((fl, ss, len(g), len(crops), n_files, len(crops & multi) / len(crops), len(crops & frag_any) / len(crops),
                      len(crops & frag8) / len(crops), len(crops & (frag_gt8 - frag8)) / len(crops)))
sd = pd.DataFrame(sess_rows, columns=["fold", "session", "boxes", "crops", "files", "multi_frac", "frag_any_frac", "frag8_frac", "frag_gt8_only_frac"])
sd = sd.sort_values(["fold", "boxes"], ascending=[True, False])
sd.to_csv(OUT / "pipe_band_v2_session_fold.csv", index=False)
P(sd.round(2).to_string(index=False))
P("\n== per-fold concentration")
fo = sd.groupby("fold").agg(sessions=("session", "size"), boxes=("boxes", "sum"), crops=("crops", "sum"), files=("files", "sum"),
                            top_boxes=("boxes", "max"))
fo["top_share"] = fo.top_boxes / fo.boxes
# fragmented crops per fold
fo["frag_any_crops"] = [len(set(bb[bb.fold_lbl == f].crop_key) & frag_any) for f in fo.index]
fo["frag8_crops"] = [len(set(bb[bb.fold_lbl == f].crop_key) & frag8) for f in fo.index]
fo["multi_crops"] = [len(set(bb[bb.fold_lbl == f].crop_key) & multi) for f in fo.index]
P(fo.round(3).to_string())

# ---- crop adjacency / overlap in x (all manifest crops) -------------------------------------
rx = re.compile(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
cks = man.drop_duplicates("crop_key")[["crop_key", "session"]].copy()
mm = cks.crop_key.str.extract(rx)
cks["line"], cks["a"], cks["b"] = mm[0], mm[1].astype(float), mm[2].astype(float)
ov = 0
adj = 0
tot_lines = 0
for ln, g in cks.groupby("line"):
    g = g.sort_values("a")
    tot_lines += 1
    a = g.a.values
    bq = g.b.values
    for i in range(len(g) - 1):
        if a[i + 1] < bq[i]:
            ov += 1
        elif a[i + 1] == bq[i]:
            adj += 1
P("\ncrop lines", tot_lines, "overlapping consecutive crop pairs (in v1 subset)", ov, "exactly adjacent", adj)
P("crop width distribution:", (cks.b - cks.a).value_counts().head(3).to_dict())
(OUT / "pipe_band_v2_report.txt").write_text("\n".join(lines), encoding="utf-8")
