"""Show how XML <filename> differs from XML member stem / image stem."""
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
zp = ROOT / "Training/02.라벨링데이터/TL_B scan_공동_기타형.zip"
n_diff = 0
shown = 0
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        if not n.lower().endswith(".xml"):
            continue
        r = ET.fromstring(z.read(n))
        fn = r.findtext("filename")
        stem = Path(n).stem
        if fn != stem:
            n_diff += 1
            if shown < 4:
                shown += 1
                print("member:", stem, "\n field :", fn, "\n local :", r.findtext("localpath"), "\n datetime:", r.findtext("datetime"), "source:", r.findtext("source"))
print("diff", n_diff)
