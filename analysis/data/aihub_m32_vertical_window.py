"""M32 fact check 2: is the vertical window of AIHub B-scan crops fixed (does every crop contain the top)?

(a) Box y statistics per equipment/class from objects.csv (all B XML, label only).
(b) C-scan channellayer (depth index) of labelled objects per equipment/class (C scan XML).
(c) Row-mean profile (horizontally coherent part = direct wave / ground coupling) of 120 trainpool crops
    per equipment, gain 1.0 (the same picks as aihub_m32_pixel_facts.py), cross-correlated with the
    equipment median profile: lag distribution. Lag 0 for (nearly) all crops means a fixed vertical window.
(d) Gain ratio from IQR/central slope of the same triples (clipping-robust).
Single process, no new randomness (reuses picks stored in m32_pixel_facts.json).
"""
import io
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
AUD = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
MAN = Path(r"C:\workspace\gpr_detection\data\splits\aihub_yolo_v1.csv")

res = {}
# (a)
ob = pd.read_csv(AUD / "objects.csv", encoding="utf-8")
ob["equip"] = ob.filename.str[:2]
ob["cy"] = (ob.ymin + ob.ymax) / 2
q = lambda s: [round(float(s.quantile(p)), 1) for p in (0.0, 0.05, 0.5, 0.95, 1.0)]
res["B_box"] = {}
for (eq, nm), g in ob.groupby(["equip", "name"]):
    res["B_box"][f"{eq}/{nm}"] = dict(n=len(g), ymin_q=q(g.ymin), cy_q=q(g.cy), ymax_q=q(g.ymax),
                                       frac_ymin_lt4=float((g.ymin < 4).mean()), frac_ymax_gt252=float((g.ymax > 252).mean()))
res["B_box_all_ymin_min"] = float(ob.ymin.min())
res["B_box_all_ymax_max"] = float(ob.ymax.max())

# (b)
rows = []
for split, pre in (("Training", "TL_"), ("Validation", "VL_")):
    for zp in sorted((ROOT / split / "02.라벨링데이터").glob(pre + "C scan*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.endswith(".xml"):
                    continue
                r = ET.fromstring(z.read(n))
                eq = r.findtext("equipment")
                cl = int(r.findtext("channellayer"))
                for o in r.findall("object"):
                    rows.append((eq, o.findtext("name"), cl))
cs = pd.DataFrame(rows, columns=["equip", "name", "layer"])
res["C_layer"] = {}
for (eq, nm), g in cs.groupby(["equip", "name"]):
    res["C_layer"][f"{eq}/{nm}"] = dict(n=len(g), layer_q=q(g.layer))

# (c),(d)
facts = json.loads((AUD / "m32_pixel_facts.json").read_text(encoding="utf-8"))
df = pd.read_csv(MAN, encoding="utf-8")
zc = {}


def load(r):
    pre, d = ("TS_", "Training") if r.split_src == "Training" else ("VS_", "Validation")
    zp = ROOT / d / "01.원천데이터" / f"{pre}B scan_{r.category}.zip"
    if zp not in zc:
        zc[zp] = zipfile.ZipFile(zp)
    return np.asarray(Image.open(io.BytesIO(zc[zp].read(r.member))).convert("L"), dtype=np.float32)


res["lag"] = {}
res["gain_iqr"] = {}
for eq, keys in facts["picks"].items():
    profs, iq = [], {"0.25": [], "0.5": []}
    for k in keys:
        sub = df[df.crop_key == k]
        ims = {float(r.gain): load(r) for r in sub.itertuples()}
        im1 = ims[1.0]
        profs.append(im1.mean(axis=1) - im1.mean())
        s1 = np.percentile(im1, 75) - np.percentile(im1, 25)
        for g in (0.25, 0.5):
            a = ims[g]
            iq[str(g)].append(float((np.percentile(a, 75) - np.percentile(a, 25)) / max(s1, 1e-6)))
    P = np.array(profs)
    med = np.median(P, axis=0)
    lags = []
    for p in P:
        best, bl = -9, 0
        for L in range(-24, 25):
            a = p[max(0, L): 256 + min(0, L)]
            b = med[max(0, -L): 256 + min(0, -L)]
            c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else -9
            if c > best:
                best, bl = c, L
        lags.append(bl)
    lags = np.array(lags)
    res["lag"][eq] = dict(n=len(lags), frac_lag0=float((lags == 0).mean()), frac_abs_le2=float((np.abs(lags) <= 2).mean()),
                          frac_abs_le8=float((np.abs(lags) <= 8).mean()), hist={int(k): int(v) for k, v in zip(*np.unique(lags, return_counts=True))})
    res["gain_iqr"][eq] = {g: [round(float(np.median(v)), 2), round(float(np.percentile(v, 10)), 2), round(float(np.percentile(v, 90)), 2)] for g, v in iq.items()}
(AUD / "m32_vertical_window.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("lag", "gain_iqr", "B_box_all_ymin_min", "B_box_all_ymax_max")}, ensure_ascii=False))
for k, v in res["B_box"].items():
    print("B", k, v)
for k, v in res["C_layer"].items():
    print("C", k, v)
