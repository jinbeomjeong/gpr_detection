"""Probe AIHub 71327 zip structure (read-only): list members and print one sample XML."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
for p in [ROOT / "Training/02.라벨링데이터/TL_B scan_비공동_종배관.zip",
          ROOT / "Training/01.원천데이터/TS_B scan_비공동_종배관.zip"]:
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        print(p.name, len(names))
        for n in names[:4]:
            print("  ", n)
        if p.name.startswith("TL_"):
            xmls = [n for n in names if n.lower().endswith(".xml")]
            print(z.read(xmls[0]).decode("utf-8", "replace"))
