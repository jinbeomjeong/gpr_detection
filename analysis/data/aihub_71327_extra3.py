"""Compare session key from XML <source> vs from XML member stem; recompute Train/Val overlap with each."""
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
rows = []
for split, pre in (("Training", "TL_"), ("Validation", "VL_")):
    for zp in sorted((ROOT / split / "02.라벨링데이터").glob(f"{pre}B scan*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.lower().endswith(".xml"):
                    continue
                r = ET.fromstring(z.read(n))
                stem = Path(n).stem
                m = re.match(r"([A-Z]{2})_(\d{4})(\d{2})(\d{2})\((\d{2})(\d{2})(\d{2})\)", stem)
                stem_sess = f"{m[1]}_{m[2]}-{m[3]}-{m[4]}_({m[5]}_{m[6]}_{m[7]})" if m else None
                rows.append({"split": split, "zip": zp.name, "stem": stem, "field": r.findtext("filename"),
                             "source": r.findtext("source"), "stem_sess": stem_sess})
D = pd.DataFrame(rows)
D["field_ne_stem"] = D.field != D.stem
D["src_ne_stem_sess"] = D.source != D.stem_sess
print("total", len(D), "stem unparsed", D.stem_sess.isna().sum())
print(D.groupby("split")[["field_ne_stem", "src_ne_stem_sess"]].sum())
print(D.groupby("zip")[["field_ne_stem", "src_ne_stem_sess"]].sum())
for key in ("source", "stem_sess"):
    a = set(D[D.split == "Training"][key]); b = set(D[D.split == "Validation"][key])
    print(key, "train", len(a), "val", len(b), "overlap", len(a & b))
print("stem unique train/val overlap:", len(set(D[D.split=='Training'].stem) & set(D[D.split=='Validation'].stem)))
D.to_csv(OUT / "stem_vs_source.csv", index=False, encoding="utf-8-sig")
