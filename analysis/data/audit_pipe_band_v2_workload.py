"""Relabel workload tiers per session/fold/test + merge-rule simulation with height constraint + line-mosaic size."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "analysis" / "data" / "out"
b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1]
man = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv", dtype={"fold": "string"})
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def merge_boxes(rows, G, hmax_ratio=None, hmax_abs=None):
    boxes = [list(r) for r in rows]
    blocked = 0
    changed = True
    while changed:
        changed = False
        order = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                ih = min(a[3], c[3]) - max(a[1], c[1])
                hmin = min(a[3] - a[1], c[3] - c[1])
                gap = max(a[0], c[0]) - min(a[2], c[2])
                if ih / hmin >= 0.5 and gap <= G:
                    order.append((gap, i, j))
        for gap, i, j in sorted(order):
            a, c = boxes[i], boxes[j]
            u = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
            uh = u[3] - u[1]
            hm = max(a[3] - a[1], c[3] - c[1])
            if (hmax_ratio and uh > hmax_ratio * hm) or (hmax_abs and uh > hmax_abs):
                blocked += 1
                continue
            boxes[i] = u
            boxes.pop(j)
            changed = True
            break
    return boxes, blocked


cb = {k: g[["x1", "y1", "x2", "y2"]].values.tolist() for k, g in bb.groupby("crop_key")}
tot0 = sum(len(v) for v in cb.values())
P("band boxes", tot0, "crops", len(cb))
for name, kw in [("G8", dict(G=8)), ("G16", dict(G=16)), ("G32", dict(G=32)),
                 ("G32+h<=1.5x", dict(G=32, hmax_ratio=1.5)), ("G32+h<=1.25x", dict(G=32, hmax_ratio=1.25)),
                 ("G16+h<=1.5x", dict(G=16, hmax_ratio=1.5))]:
    n = 0
    blk = 0
    chg = 0
    for k, v in cb.items():
        m, bl = merge_boxes(v, **kw)
        n += len(m)
        blk += bl
        chg += int(len(m) != len(v))
    P(f"{name}: boxes {n} ({(n-tot0)/tot0*100:+.1f}%), crops changed {chg}, blocked-by-height {blk}")
# heights: how tall do union boxes get under G32
h_all = bb.h
P("band box h p50/p95/max:", h_all.quantile([.5, .95, 1]).round(1).tolist())

# tiers per crop
def pair_info(rows):
    gap_any = gap_small = False
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, c = rows[i], rows[j]
            ih = min(a[3], c[3]) - max(a[1], c[1])
            if ih / min(a[3] - a[1], c[3] - c[1]) >= 0.5:
                gap = max(a[0], c[0]) - min(a[2], c[2])
                if gap > 0:
                    gap_any = True
                    if gap <= 8:
                        gap_small = True
    return gap_any, gap_small


crop_df = bb.drop_duplicates("crop_key")[["crop_key", "session", "fold_lbl"]].set_index("crop_key")
tier = {}
for k, v in cb.items():
    ga, gs = pair_info(v)
    n = len(v)
    touch = bool(bb[bb.crop_key == k].touch_x.any()) if False else None
    tier[k] = "A_gapped" if ga else ("B_multi_nogap" if n >= 2 else "C_single")
crop_df["tier"] = pd.Series(tier)
edge = bb.groupby("crop_key").touch_x.any()
crop_df["edge"] = edge
nfiles = man.groupby("crop_key").size()
crop_df["files"] = nfiles
rows = []
for (f, s), g in crop_df.groupby(["fold_lbl", "session"]):
    rows.append((f, s, len(g), int(g.files.sum()), (g.tier == "A_gapped").sum(), (g.tier == "B_multi_nogap").sum(), (g.tier == "C_single").sum(),
                 int(((g.tier == "C_single") & g.edge).sum())))
t = pd.DataFrame(rows, columns=["fold", "session", "crops", "files", "A_gapped", "B_multi", "C_single", "C_single_edge"])
t = t.sort_values(["fold", "crops"], ascending=[True, False])
P(t.to_string(index=False))
tot = t[["crops", "files", "A_gapped", "B_multi", "C_single", "C_single_edge"]].sum()
P("TOTAL", tot.to_dict())
P("by fold:")
P(t.groupby("fold")[["crops", "files", "A_gapped", "B_multi", "C_single"]].sum().to_string())
t.to_csv(OUT / "pipe_band_v2_workload_tiers.csv", index=False)
crop_df.to_csv(OUT / "pipe_band_v2_crop_tiers.csv")

# line-mosaic size
rx = re.compile(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
allc = man.drop_duplicates("crop_key")[["crop_key", "session", "subset", "fold"]].copy()
m = allc.crop_key.str.extract(rx)
allc["line"], allc["a"], allc["b"] = m[0], m[1].astype(int), m[2].astype(int)
band_keys = set(cb)
lines_with_band = set(allc[allc.crop_key.isin(band_keys)].line)


def union_len(iv):
    iv = sorted(iv)
    tot = 0
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            tot += cur_e - cur_s
            cur_s, cur_e = s, e
    return tot + cur_e - cur_s


bandc = allc[allc.crop_key.isin(band_keys)]
u_band = sum(union_len(list(zip(g.a, g.b))) for _, g in bandc.groupby("line"))
u_all = sum(union_len(list(zip(g.a, g.b))) for ln, g in allc[allc.line.isin(lines_with_band)].groupby("line"))
P(f"\nband crops {len(bandc)} (x256 = {len(bandc)*256} px), union over band crops {u_band} px = {u_band/256:.0f} crop-equivalents, "
  f"lines {bandc.line.nunique()}; union incl. non-band v1 crops on same lines {u_all} px = {u_all/256:.0f} eq")
# contiguous segments (islands) of band crops per line, for mosaic windows
segs = []
for ln, g in bandc.groupby("line"):
    iv = sorted(zip(g.a, g.b))
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            segs.append((ln, cs, ce))
            cs, ce = s, e
    segs.append((ln, cs, ce))
sg = pd.DataFrame(segs, columns=["line", "s", "e"])
sg["len"] = sg.e - sg.s
P("contiguous islands:", len(sg), "len p50/p90/max px:", sg.len.quantile([.5, .9, 1]).astype(int).tolist(), "islands single crop-sized (<=256):", int((sg.len <= 256).sum()))
(OUT / "pipe_band_v2_workload_report.txt").write_text("\n".join(lines), encoding="utf-8")
