"""Extra checks: label/image name mismatches, equipment per class, sessions per class by equipment."""
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
F = pd.read_csv(OUT / "files.csv")
O = pd.read_csv(OUT / "objects.csv")
zp = ROOT / "Training/02.라벨링데이터/TL_B scan_공동_기타형.zip"
with zipfile.ZipFile(ROOT / "Training/01.원천데이터/TS_B scan_공동_기타형.zip") as z:
    imgs = {Path(n).stem for n in z.namelist() if n.lower().endswith(".jpg")}
with zipfile.ZipFile(zp) as z:
    xml_stems = {Path(n).stem for n in z.namelist() if n.lower().endswith(".xml")}
labs = set(F[F.zip == zp.name].filename)
print("xml members", len(xml_stems), "xml filename fields", len(labs), "images", len(imgs))
print("stem xml-img only", sorted(xml_stems - imgs)[:3], len(xml_stems - imgs))
print("filename-field not in imgs", sorted(labs - imgs)[:3], len(labs - imgs))
print("filename-field not in xml stems", sorted(labs - xml_stems)[:3], len(labs - xml_stems))
print("duplicate filename fields:", F.duplicated(["split", "filename"]).sum())
print(O.merge(F[["split", "filename", "equipment", "gainvalue"]], on=["split", "filename"]).groupby(["name", "equipment"]).agg(obj=("name", "size"), sess=("source", "nunique")))
print(F.groupby("zip").agg(files=("filename", "size"), sess=("source", "nunique")))
print("sessions with >1 equipment:", (F.groupby("source").equipment.nunique() > 1).sum())
print("source examples", F.source.drop_duplicates().sample(5, random_state=1).tolist())
