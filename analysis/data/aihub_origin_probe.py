"""Probe AIHub 71327 source zips: list members (no extraction), sample image sizes/names.
Read-only. Usage: python -I aihub_origin_probe.py
"""
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = Path(r"C:\workspace\gpr_detection\analysis\data\aihub_origin_probe.txt")
lines = []


def log(s):
    print(s)
    lines.append(s)


for split in ("Training", "Validation"):
    for z in sorted((ROOT / split / "01.원천데이터").glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            infos = zf.infolist()
            files = [i for i in infos if not i.is_dir()]
            ext = Counter(Path(i.filename).suffix.lower() for i in files)
            pat = Counter()
            for i in files:
                n = Path(i.filename).name
                pat[re.sub(r"\d+", "#", n)[:60]] += 1
            log(f"== {split}/{z.name}: files={len(files)} ext={dict(ext)}")
            log(f"   top-dirs={Counter(i.filename.strip('/').split('/')[0] for i in files).most_common(3)}")
            log(f"   first={[i.filename for i in files[:2]]}")
            log(f"   patterns={pat.most_common(3)}")
            # sample sizes
            sizes = Counter()
            for i in files[:: max(1, len(files) // 25)]:
                if i.filename.lower().endswith((".jpg", ".png")):
                    im = Image.open(io.BytesIO(zf.read(i)))
                    sizes[(im.size, im.mode)] += 1
            log(f"   sample (size,mode)={dict(sizes)}")
OUT.write_text("\n".join(lines), encoding="utf-8")
