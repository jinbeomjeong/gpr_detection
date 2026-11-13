"""Context availability if ALL B-scan crops in the source zips (74,838, listing only) are used as neighbours,
not only the 16,887 images in manifest v1. Read-only: zip central directory only, no extraction.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
SRC = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = ROOT / "data" / "aihub_context_index"
rx = re.compile(r"^/?(.*_N\d+_B\d+)_g([\d.]+)_x\((\d+)-(\d+)\)\.jpg$")

rows = []
for split, pre in (("Training", "TS"), ("Validation", "VS")):
    for z in sorted((SRC / split / "01.원천데이터").glob(f"{pre}_B scan_*.zip")):
        cat = z.stem.split("_", 2)[2]
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.is_dir():
                    continue
                m = rx.match(i.filename)
                assert m, i.filename
                rows.append((m[1], float(m[2]), int(m[3]), int(m[4]), cat, split))
full = pd.DataFrame(rows, columns=["line", "gain", "a", "b", "category", "split_src"])
print("full B members", len(full), "unique (line,gain,a,b)", len(full.drop_duplicates(["line", "gain", "a", "b"])))
print("unique crops (no gain)", len(full.drop_duplicates(["line", "a", "b"])))
uf = full.drop_duplicates(["line", "gain", "a", "b"]).copy()

man = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv", dtype={"fold": "string"})
m = man.crop_key.str.extract(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
man["line"], man["a"], man["b"] = m[0], m[1].astype(int), m[2].astype(int)
man["gain"] = man.gain.astype(float)
man_lines = set(man.line)
print("lines full", uf.line.nunique(), "lines in manifest", len(man_lines), "full lines not in manifest", len(set(uf.line) - man_lines))
print("manifest rows in full listing:", len(man.merge(uf, on=["line", "gain", "a", "b"])), "of", len(man))
extra = uf[uf.line.isin(man_lines)].merge(man[["line", "gain", "a", "b"]].assign(inman=1), on=["line", "gain", "a", "b"], how="left")
print("full crops on manifest lines:", len(extra), " of which not in manifest:", int(extra.inman.isna().sum()))
print("extra (non-manifest) crops on manifest lines by category:\n", extra[extra.inman.isna()].category.value_counts().to_string())
print("full crops on lines NOT in manifest by category:\n", uf[~uf.line.isin(man_lines)].category.value_counts().to_string())

# context extension with full set (same line, same gain); chain over contiguous union
by = {k: g[["a", "b"]].values for k, g in uf.groupby(["line", "gain"])}

def ext(line, gain, a, b):
    iv = by[(line, gain)]
    lo, hi = a, b
    ch = True
    while ch:
        ch = False
        for x, y in iv:
            if x <= hi and y >= lo and (x < lo or y > hi):
                lo, hi, ch = min(lo, x), max(hi, y), True
    return a - lo, hi - b

res = man.apply(lambda r: ext(r.line, r.gain, r.a, r.b), axis=1, result_type="expand")
man["fL"], man["fR"] = res[0], res[1]
man["fmax"] = man[["fL", "fR"]].max(axis=1)
man["fmin"] = man[["fL", "fR"]].min(axis=1)
man[["file_name", "crop_key", "subset", "fold", "primary_class", "fL", "fR"]].to_csv(OUT / "crops_fullset_ext.csv", index=False)
for name, sub in (("trainpool", man[man.subset == "trainpool"]), ("test", man[man.subset == "test"])):
    out = []
    for cls, s in sub.groupby("primary_class"):
        out.append(dict(cls=cls, crops=len(s), any=round((s.fmax > 0).mean() * 100, 1), ge32=round((s.fmax >= 32).mean() * 100, 1),
                        ge64=round((s.fmax >= 64).mean() * 100, 1), ge96=round((s.fmax >= 96).mean() * 100, 1),
                        both64=round((s.fmin >= 64).mean() * 100, 1), median_ext_when_any=int(s.fmax[s.fmax > 0].median()) if (s.fmax > 0).any() else 0))
    print(f"\n== fullset context, {name}"); print(pd.DataFrame(out).to_string(index=False))
