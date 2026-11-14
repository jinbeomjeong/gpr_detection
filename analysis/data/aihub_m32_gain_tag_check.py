"""M32 fact check 1b: XML gainvalue vs file-name g<gain> tag; counts of gain triples (B scan XML, all 74,838)."""
import re
from pathlib import Path

import pandas as pd

f = pd.read_csv(r"C:\workspace\gpr_detection\data\audit\aihub_71327\files.csv", encoding="utf-8")
f["tag"] = f.filename.str.extract(r"_g([0-9.]+)_x")[0].astype(float)
print("n", len(f), "tag null", int(f.tag.isna().sum()))
print("gainvalue != name tag:", int((f.gainvalue.astype(float) != f.tag).sum()))
f["crop"] = f.filename.str.replace(r"_g[0-9.]+_x", "_x", regex=True)
g = f.groupby("crop").gainvalue.nunique()
print("crops by n_gains:", g.value_counts().sort_index().to_dict())
print("gain x equipment:\n", pd.crosstab(f.equipment, f.gainvalue))
