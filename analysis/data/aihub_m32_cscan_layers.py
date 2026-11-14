"""M32 fact check: C-scan channellayer range per equipment (= depth-layer index) from label XML only.

Reads TL_/VL_ C scan label zips with zipfile (no extraction). Single process, no randomness.
Also tallies XML tag names so that any gain/processing-history field is visible.
"""
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327\m32_cscan_layers.json")

layers = defaultdict(lambda: defaultdict(set))  # equipment -> scanclass -> set(layer)
sizes = defaultdict(Counter)
tags = Counter()
n_xml = Counter()
for split, pre in (("Training", "TL_"), ("Validation", "VL_")):
    for zp in sorted((ROOT / split / "02.라벨링데이터").glob(pre + "*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith(".xml"):
                    continue
                root = ET.fromstring(z.read(name))
                for ch in root:
                    tags[ch.tag] += 1
                eq = root.findtext("equipment")
                sc = root.findtext("scanclass")
                cl = int(root.findtext("channellayer"))
                w = root.findtext("size/width")
                h = root.findtext("size/height")
                layers[eq][sc].add(cl)
                sizes[(eq, sc)][(w, h)] += 1
                n_xml[sc] += 1

res = {"n_xml": dict(n_xml), "tags": dict(tags), "layers": {}, "sizes": {}}
for eq, d in layers.items():
    for sc, s in d.items():
        s = sorted(s)
        res["layers"][f"{eq}/{sc}"] = {"min": s[0], "max": s[-1], "n_distinct": len(s)}
for k, c in sizes.items():
    res["sizes"][f"{k[0]}/{k[1]}"] = {f"{a}x{b}": n for (a, b), n in c.items()}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
