"""Reproduce multi-group counts per class (read-only on index CSVs)."""
from pathlib import Path

import pandas as pd

D = Path(r"C:\workspace\gpr_detection\data\aihub_context_index")
c = pd.read_csv(D / "crops.csv")
g = pd.read_csv(D / "groups.csv")
t = c[c.subset == "trainpool"]
gt = g[(g.subset == "trainpool") & (g.n > 1)]
print("groups.csv trainpool n>1:", len(gt), gt.classes.value_counts().to_dict())
for cls in ("pipe_band", "pipe_hyperbola", "cavity", "manhole"):
    s = t[t.primary_class == cls].groupby("group_id").size()
    print(cls, "groups with >=2 crops of this class:", int((s > 1).sum()),
          " groups n>1 containing this class:", int(gt.classes.str.contains(cls).sum()))
