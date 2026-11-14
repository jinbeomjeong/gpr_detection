"""M32 fact check 3: is there a common horizontal (row-mean) pattern shared by crops of different sessions?

If a per-line background/mean-trace subtraction had been applied, the signed row-mean profile of a crop
would be unrelated to other sessions (common-mode fraction ~ 1/N). A persistent system pattern
(direct wave, ringing) that survives in many sessions gives a common-mode fraction well above 1/N.
One gain-1.0 trainpool crop per session (random, seed 20261114), up to 300 sessions per equipment.
Common-mode fraction per row band = sum_r (mean_i p_i[r])^2 / sum_r mean_i (p_i[r]^2), p_i = row mean - crop mean.
Also prints the per-row sign agreement with the equipment median profile.
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
AUD = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
MAN = Path(r"C:\workspace\gpr_detection\data\splits\aihub_yolo_v1.csv")
SEED = 20261114
NMAX = 300

df = pd.read_csv(MAN, encoding="utf-8")
df = df[(df.subset == "trainpool") & (df.gain == 1.0)].copy()
df["equip"] = df.file_name.str[:2]
rng = np.random.default_rng(SEED)
zc = {}


def load(r):
    pre, d = ("TS_", "Training") if r.split_src == "Training" else ("VS_", "Validation")
    zp = ROOT / d / "01.원천데이터" / f"{pre}B scan_{r.category}.zip"
    if zp not in zc:
        zc[zp] = zipfile.ZipFile(zp)
    return np.asarray(Image.open(io.BytesIO(zc[zp].read(r.member))).convert("L"), dtype=np.float32)


res = {}
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
bands = {"0-31": (0, 32), "32-95": (32, 96), "96-255": (96, 256), "all": (0, 256)}
for eq in ["DS", "MA", "NM", "RA"]:
    sub = df[df.equip == eq]
    sess = sorted(sub.session.unique())
    sel = list(rng.choice(sess, size=min(NMAX, len(sess)), replace=False))
    P = []
    for s in sel:
        r = sub[sub.session == s].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        im = load(r)
        P.append(im.mean(axis=1) - im.mean())
    P = np.array(P)
    mean_p = P.mean(axis=0)
    out = {"n_sessions": len(P)}
    for name, (a, b) in bands.items():
        cm = float((mean_p[a:b] ** 2).sum() / (P[:, a:b] ** 2).mean(axis=0).sum())
        out[f"common_mode_{name}"] = round(cm, 4)
    out["one_over_N"] = round(1 / len(P), 4)
    med = np.median(P, axis=0)
    sign_agree = (np.sign(P) == np.sign(med)).mean(axis=0)
    out["sign_agree_rows0-31_median"] = round(float(np.median(sign_agree[:32])), 3)
    out["sign_agree_rows32-255_median"] = round(float(np.median(sign_agree[32:])), 3)
    out["peak_abs_mean_row"] = int(np.argmax(np.abs(mean_p)))
    out["mean_profile_first40"] = mean_p[:40].round(1).tolist()
    out["profile_std_over_sessions_first40"] = P.std(axis=0)[:40].round(1).tolist()
    res[eq] = out
    ax.plot(mean_p, label=f"{eq} (n={len(P)})")
ax.set_xlabel("row")
ax.set_ylabel("mean over sessions of (row mean - crop mean), gain 1.0")
ax.legend()
fig.tight_layout()
fig.savefig(AUD / "m32_common_mode_profile.png", dpi=80)
(AUD / "m32_common_mode.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for eq, o in res.items():
    print(eq, {k: v for k, v in o.items() if not k.startswith(("mean_profile", "profile_std"))})
    print("  mean_profile", o["mean_profile_first40"][:40])
