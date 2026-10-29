"""Print the audit tables (markdown-ish) for aihub_71327_audit.py outputs."""
import json
from pathlib import Path
import pandas as pd

OUT = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 200)
for n in ["pairing", "class_all", "class_by_split"]:
    print("==", n); print(pd.read_csv(OUT / f"{n}.csv").to_string(index=False))
for f in ["equipment", "channels", "gainvalue", "permittivity", "scanclass", "channellayer", "roadtype"]:
    print("==", f); print(pd.read_csv(OUT / f"dist_{f}.csv").to_string(index=False))
print(json.dumps(json.load(open(OUT / "overlap.json")), indent=1))
F = pd.read_csv(OUT / "files.csv")
print(F.groupby("split").agg(files=("filename", "size"), sessions=("source", "nunique"), nobj0=("nobj", lambda s: (s == 0).sum())))
print(F.groupby(["w", "h"]).size())
