"""M32 fact check: AIHub gain triples and the vertical (row) structure of B-scan crops.

Reads trainpool JPGs straight from the TS_/VS_ B scan zips (no extraction, test set untouched).
Seed 20261114, single process. Samples N_PER_EQ gain triples (same crop_key, same x) per equipment.
Outputs: data/audit/aihub_71327/m32_pixel_facts.json, m32_gain_triples.png, m32_row_profile.png
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
MAN = Path(r"C:\workspace\gpr_detection\data\splits\aihub_yolo_v1.csv")
OUTD = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
SEED = 20261114
N_PER_EQ = 120

df = pd.read_csv(MAN, encoding="utf-8")
df = df[df.subset == "trainpool"].copy()
df["equip"] = df.file_name.str[:2]
cnt = df.groupby("crop_key").gain.nunique()
full = cnt[cnt == 3].index
dtri = df[df.crop_key.isin(full)]
rng = np.random.default_rng(SEED)

zcache = {}


def load(row):
    prefix = "TS_" if row.split_src == "Training" else "VS_"
    d = "Training" if row.split_src == "Training" else "Validation"
    zp = ROOT / d / "01.원천데이터" / f"{prefix}B scan_{row.category}.zip"
    if zp not in zcache:
        zcache[zp] = zipfile.ZipFile(zp)
    b = zcache[zp].read(row.member)
    return np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=np.float32)


res = {"seed": SEED, "n_per_eq": N_PER_EQ, "eq": {}}
prof = {}  # (eq, gain) -> list of row-energy arrays
rowmean = {}
picks = {}
for eq in ["DS", "MA", "NM", "RA"]:
    keys = sorted(dtri[dtri.equip == eq].crop_key.unique())
    sel = list(rng.choice(keys, size=min(N_PER_EQ, len(keys)), replace=False))
    picks[eq] = [str(k) for k in sel]
    stats = {g: [] for g in (0.25, 0.5, 1.0)}
    rel = {"0.25_vs_1.0": [], "0.5_vs_1.0": []}
    shown = []
    for k in sel:
        sub = dtri[dtri.crop_key == k].set_index("gain")
        imgs = {g: load(sub.loc[g].iloc[0] if isinstance(sub.loc[g], pd.DataFrame) else sub.loc[g]) for g in (0.25, 0.5, 1.0)}
        if len(shown) < 3:
            shown.append(imgs)
        for g, im in imgs.items():
            m, s = im.mean(), im.std()
            e_row = ((im - m) ** 2).mean(axis=1)  # row energy
            rm = im.mean(axis=1)
            stats[g].append(
                dict(
                    mean=float(m), std=float(s), p1=float(np.percentile(im, 1)), p99=float(np.percentile(im, 99)),
                    lo=float((im <= 2).mean()), hi=float((im >= 253).mean()),
                    peak_row=int(np.argmax(e_row)), top8_over_all=float(e_row[:8].mean() / max(e_row.mean(), 1e-9)),
                    top16_over_all=float(e_row[:16].mean() / max(e_row.mean(), 1e-9)),
                    bottom32_over_all=float(e_row[-32:].mean() / max(e_row.mean(), 1e-9)),
                )
            )
            prof.setdefault((eq, g), []).append(e_row)
            rowmean.setdefault((eq, g), []).append(rm)
        b = imgs[1.0].ravel()
        for g, name in ((0.25, "0.25_vs_1.0"), (0.5, "0.5_vs_1.0")):
            a = imgs[g].ravel()
            sl, ic = np.polyfit(b, a, 1)
            r = float(np.corrcoef(a, b)[0, 1])
            # conditional mean of a over 9 quantile bins of b
            qs = np.quantile(b, np.linspace(0, 1, 10))
            idx = np.clip(np.searchsorted(qs, b, side="right") - 1, 0, 8)
            cm = [float(a[idx == i].mean()) for i in range(9)]
            bm = [float(b[idx == i].mean()) for i in range(9)]
            rel[name].append(dict(slope=float(sl), icpt=float(ic), r=r, cond_a=cm, cond_b=bm))
    out = {}
    for g in stats:
        t = pd.DataFrame(stats[g])
        out[str(g)] = {c: [float(t[c].median()), float(t[c].quantile(0.1)), float(t[c].quantile(0.9))] for c in t.columns if c != "peak_row"}
        out[str(g)]["peak_row_hist16"] = np.bincount(t.peak_row.values // 16, minlength=16).tolist()
        out[str(g)]["peak_row_median"] = float(t.peak_row.median())
    for name, lst in rel.items():
        t = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("cond")} for d in lst])
        out[name] = {c: [float(t[c].median()), float(t[c].quantile(0.1)), float(t[c].quantile(0.9))] for c in t.columns}
        out[name]["cond_a_median"] = np.median([d["cond_a"] for d in lst], axis=0).round(1).tolist()
        out[name]["cond_b_median"] = np.median([d["cond_b"] for d in lst], axis=0).round(1).tolist()
    res["eq"][eq] = out
    # triple figure
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))
    for i, imgs in enumerate(shown):
        for j, g in enumerate((0.25, 0.5, 1.0)):
            ax[i, j].imshow(imgs[g], cmap="gray", vmin=0, vmax=255)
            ax[i, j].set_title(f"{eq} g{g}", fontsize=8)
            ax[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(OUTD / f"m32_gain_triples_{eq}.png", dpi=80)
    plt.close(fig)

# row-energy profiles (mean over crops, normalized by each crop's overall mean energy)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
rp = {}
for (eq, g), lst in prof.items():
    a = np.array(lst)
    a = a / a.mean(axis=1, keepdims=True)
    mp = a.mean(axis=0)
    rp[f"{eq}_g{g}"] = mp[:48].round(2).tolist()
    if g == 1.0:
        ax[0].plot(mp, label=eq)
    if eq == "NM":
        ax[1].plot(mp, label=f"NM g{g}")
ax[0].set_title("normalized row energy, gain 1.0")
ax[1].set_title("normalized row energy, NM per gain")
for a_ in ax:
    a_.legend()
    a_.set_xlabel("row")
fig.tight_layout()
fig.savefig(OUTD / "m32_row_profile.png", dpi=80)
res["row_energy_first48"] = rp
res["picks"] = picks
(OUTD / "m32_pixel_facts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("done")
