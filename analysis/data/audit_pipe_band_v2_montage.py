"""pipe_band audit v2 - visual review montages (local only: contains AIHub images, git-ignored dirs).

(1) 30 same-depth band box pairs with x gap > 8 px (10 each in 8-16 / 16-32 / >32 px), session round-robin, seed 20261031.
(2) 25 'no_gt' FP boxes that sit in the same depth row as a GT band box (M1-7 fold 0, last.pt seed 0).
Adds a horizontal-coherence metric V (std over rows of the x-averaged row profile) for gap region vs boxes vs control.
Outputs: docs/datasets/figures/pipe_band_shape/pipe_band_v2_*.png, analysis/data/out/pipe_band_v2_gap_sample.csv, pipe_band_v2_gap_metric.csv
"""
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
FIG.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "data" / "aihub_yolo_v1" / "images"
SEED = 20261031
rng = np.random.default_rng(SEED)

b = pd.read_csv(OUT / "pipe_band_boxes.csv")
bb = b[b.cls == 1]
file_of = bb.drop_duplicates("crop_key").set_index("crop_key")["file"].to_dict()
boxes_of = {ck: g[["x1", "y1", "x2", "y2"]].values.tolist() for ck, g in bb.groupby("crop_key")}
cache = {}


def load(ck):
    if ck not in cache:
        cache[ck] = np.asarray(Image.open(IMG / file_of[ck]).convert("L"), dtype=np.float32) / 255.0
    return cache[ck]


def vcoh(im, x1, x2, y1, y2, pad=6):
    """std over rows of the x-averaged row profile in rows [y1-pad, y2+pad] (horizontal coherence)."""
    x1, x2 = int(round(max(x1, 0))), int(round(min(x2, 256)))
    y1, y2 = int(round(max(y1 - pad, 0))), int(round(min(y2 + pad, 256)))
    if x2 - x1 < 3 or y2 - y1 < 4:
        return np.nan
    prof = im[y1:y2, x1:x2].mean(axis=1)
    return float(prof.std())


gp = pd.read_csv(OUT / "pipe_band_v2_gap_gt8_pairs.csv")
def is_adjacent(r):
    """True if no third band box lies (in x) inside the gap and shares the depth row with either box."""
    A = (r.ax1, r.ay1, r.ax2, r.ay2)
    Bq = (r.bx1, r.by1, r.bx2, r.by2)
    gl, gr = min(A[2], Bq[2]), max(A[0], Bq[0])
    for c in boxes_of[r.crop_key]:
        if np.allclose(c, A, atol=1e-3) or np.allclose(c, Bq, atol=1e-3):
            continue
        if c[0] < gr and c[2] > gl:
            for o in (A, Bq):
                ih = min(c[3], o[3]) - max(c[1], o[1])
                if ih / min(c[3] - c[1], o[3] - o[1]) >= 0.5:
                    return False
    return True


gp["adjacent"] = [is_adjacent(r) for r in gp.itertuples()]
gp["bin"] = pd.cut(gp.xgap, [8, 16, 32, 300], labels=["8-16", "16-32", "gt32"])
print("all same-depth gap>8 pairs", len(gp), "adjacent-only", int(gp.adjacent.sum()))
print(gp.groupby(["bin", "adjacent"], observed=True).size().unstack())
print(gp[gp.adjacent].groupby("bin", observed=True).crop_key.nunique())
gp.to_csv(OUT / "pipe_band_v2_gap_gt8_pairs_adj.csv", index=False)
gp = gp[gp.adjacent]
# one pair per crop: the smallest gap > 8 in the crop
gp = gp.sort_values("xgap").drop_duplicates("crop_key")
gp["bin"] = pd.cut(gp.xgap, [8, 16, 32, 300], labels=["8-16", "16-32", "gt32"])
print(gp.groupby(["bin", "session"], observed=True).size())
sel = []
for bn in ["8-16", "16-32", "gt32"]:
    g = gp[gp.bin == bn]
    per = {s: x.sample(frac=1, random_state=SEED).to_dict("records") for s, x in g.groupby("session")}
    picked = []
    while len(picked) < 10 and any(per.values()):
        for s in list(per):
            if per[s] and len(picked) < 10:
                picked.append(per[s].pop())
    sel += picked
sel = pd.DataFrame(sel)

# metric for all gap>8 pairs (not just the sample)
rows = []
for r in gp.itertuples():
    im = load(r.crop_key)
    lo, hi = sorted([(r.ax1, r.ax2, r.ay1, r.ay2), (r.bx1, r.bx2, r.by1, r.by2)], key=lambda t: t[0])
    gx1, gx2 = lo[1], hi[0]
    y1 = min(lo[2], hi[2])
    y2 = max(lo[3], hi[3])
    va = vcoh(im, lo[0], lo[1], lo[2], lo[3])
    vb = vcoh(im, hi[0], hi[1], hi[2], hi[3])
    vg = vcoh(im, gx1, gx2, y1, y2)
    # control: same-width window at same rows, outside all GT band boxes' x-range, if any room
    w = gx2 - gx1
    vc = np.nan
    occ = [(bx[0], bx[2]) for bx in boxes_of[r.crop_key]]
    for _ in range(40):
        cx = rng.uniform(0, 256 - w)
        if all(cx + w <= o[0] or cx >= o[1] for o in occ):
            vc = vcoh(im, cx, cx + w, y1, y2)
            break
    rows.append((r.crop_key, r.session, r.fold_lbl, r.xgap, va, vb, vg, vc))
met = pd.DataFrame(rows, columns=["crop_key", "session", "fold_lbl", "xgap", "v_a", "v_b", "v_gap", "v_ctrl"])
met["v_box"] = met[["v_a", "v_b"]].mean(axis=1)
met["r_gap"] = met.v_gap / met.v_box
met["r_ctrl"] = met.v_ctrl / met.v_box
met["bin"] = pd.cut(met.xgap, [8, 16, 32, 300], labels=["8-16", "16-32", "gt32"])
met.to_csv(OUT / "pipe_band_v2_gap_metric.csv", index=False)
print(met.groupby("bin", observed=True)[["r_gap", "r_ctrl"]].agg(["count", "median", "mean"]).round(2))
sel = sel.merge(met[["crop_key", "r_gap", "r_ctrl"]], on="crop_key", how="left")
sel.to_csv(OUT / "pipe_band_v2_gap_sample.csv", index=False)


def tile(ax, ck, pair=None, extra=None, title=""):
    im = load(ck)
    ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for bx in boxes_of[ck]:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="#00e5ff", lw=0.8))
    if pair:
        for (x1, y1, x2, y2), c in zip(pair, ["#ff3030", "#ffa000"]):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=1.6))
    if extra:
        ax.add_patch(Rectangle((extra[0], extra[1]), extra[2] - extra[0], extra[3] - extra[1], fill=False, ec="#ffee00", lw=1.6, ls="--"))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


for bn in ["8-16", "16-32", "gt32"]:
    s = sel[sel.bin == bn].reset_index(drop=True)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4), dpi=80)
    for ax in axs.ravel():
        ax.axis("off")
    for i, r in s.iterrows():
        ax = axs.ravel()[i]
        ax.axis("on")
        tile(ax, r.crop_key, pair=[(r.ax1, r.ay1, r.ax2, r.ay2), (r.bx1, r.by1, r.bx2, r.by2)],
             title=f"#{i+1:02d} {r.crop_key[:23]}..{r.crop_key[-14:]}\ngap={r.xgap:.0f}px  r_gap={r.r_gap:.2f} ctrl={r.r_ctrl:.2f}")
    fig.suptitle(f"pipe_band same-depth pairs, x gap {bn} px (red/orange = pair, cyan = other GT boxes) - LOCAL ONLY", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"pipe_band_v2_gap_{bn}.png")
    plt.close(fig)

# (2) no_gt FP same row
ng = pd.read_csv(OUT / "pipe_band_v2_nogt_fp.csv")
sr = ng[ng.same_row].sort_values("gap_to_gt").reset_index(drop=True)
fig, axs = plt.subplots(5, 5, figsize=(20, 20), dpi=80)
for ax in axs.ravel():
    ax.axis("off")
for i, r in sr.iterrows():
    ax = axs.ravel()[i]
    ax.axis("on")
    tile(ax, r.crop_key, extra=(r.px1, r.py1, r.px2, r.py2),
         title=f"#{i+1:02d} {r.crop_key[:23]}..{r.crop_key[-14:]}\nFP conf={r.conf:.2f} gap_to_GT={r.gap_to_gt:.0f}px")
fig.suptitle("M1-7 fold0 no_gt FP in GT-band row (yellow dashed = FP, cyan = GT band boxes) sorted by gap - LOCAL ONLY", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "pipe_band_v2_nogt_samerow.png")
plt.close(fig)
sr.assign(idx=np.arange(1, len(sr) + 1)).to_csv(OUT / "pipe_band_v2_nogt_samerow_order.csv", index=False)
print("done", len(sel), len(sr))
