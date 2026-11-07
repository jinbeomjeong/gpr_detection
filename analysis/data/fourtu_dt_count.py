"""Recount 4TU sample interval (dt_ns) per site from metadata index and repo lines csv."""
from pathlib import Path

import pandas as pd

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META = load_dataset_paths().fourtu / "01_메타데이터" / "radargrams_index.csv"
LINES = PROJECT_ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"

m = pd.read_csv(META)
print("meta rows", len(m))
print([c for c in m.columns])
print(m["dt_ns"].round(3).value_counts().sort_index())
m["site"] = m["activity_id"].astype(str).str.split(".").str[0]
print(m.groupby(["site", m["dt_ns"].round(3)]).size().unstack(fill_value=0))

l = pd.read_csv(LINES)
print("lines rows", len(l))
print(list(l.columns))
dtc = [c for c in l.columns if "dt" in c.lower()]
print(dtc)
if dtc:
    print(l[dtc[0]].round(3).value_counts().sort_index())
    sc = [c for c in l.columns if "site" in c.lower()]
    if sc:
        print(l.groupby([sc[0], l[dtc[0]].round(3)]).size().unstack(fill_value=0))
