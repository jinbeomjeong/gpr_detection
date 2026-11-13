"""Build overlapping/contiguous crop groups (same session, line(channel), gain) from the AIHub split manifest.
Read-only on manifest. Outputs data/aihub_context_index/{crops,groups}.csv and prints summary tables.
Manifest: data/splits/aihub_yolo_v1.csv (no zip access needed: x range is in the file name).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\workspace\gpr_detection")
OUT = ROOT / "data" / "aihub_context_index"
OUT.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(ROOT / "data" / "splits" / "aihub_yolo_v1.csv", dtype={"fold": "string"})
print("rows", len(man), "subset", man.subset.value_counts().to_dict(), "fold", man.fold.value_counts(dropna=False).to_dict())
print("n_boxes min", man.n_boxes.min(), "primary_class", man.primary_class.value_counts().to_dict())

rx = re.compile(r"^(.*_N\d+_B\d+)_x\((\d+)-(\d+)\)$")
m = man.crop_key.str.extract(rx)
assert m[0].notna().all()
man["line"] = m[0]
man["a"] = m[1].astype(int)
man["b"] = m[2].astype(int)
man["gain"] = man.gain.astype(float)
print("crop width values", (man.b - man.a).value_counts().to_dict())
# the same line must map to one session / subset / fold
chk = man.groupby("line").agg(ns=("session", "nunique"), nf=("fold", "nunique"), nsub=("subset", "nunique"))
print("lines", len(chk), "lines with >1 session", int((chk.ns > 1).sum()), ">1 fold", int((chk.nf > 1).sum()), ">1 subset", int((chk.nsub > 1).sum()))

def components(g):
    """Union of intervals joined when they overlap or touch (a2 <= b1). Returns comp id per row (g sorted by a)."""
    comp = np.zeros(len(g), int)
    cur, hi = 0, -1
    for i, (a, b) in enumerate(zip(g.a.values, g.b.values)):
        if i == 0:
            hi = b
        elif a <= hi:
            hi = max(hi, b)
        else:
            cur += 1
            hi = b
        comp[i] = cur
    return comp

def direct_ext(g):
    """Context extension using only crops that directly overlap/touch this crop (no chaining)."""
    A, B = g.a.values, g.b.values
    L = np.zeros(len(g), int)
    R = np.zeros(len(g), int)
    for i in range(len(g)):
        ov = (A <= B[i]) & (B >= A[i])
        ov[i] = False
        if ov.any():
            L[i] = max(0, A[i] - A[ov].min())
            R[i] = max(0, B[ov].max() - B[i])
    return L, R

def chain_ext(g, comp):
    L = np.zeros(len(g), int)
    R = np.zeros(len(g), int)
    for c in np.unique(comp):
        idx = np.where(comp == c)[0]
        lo, hi = g.a.values[idx].min(), g.b.values[idx].max()
        L[idx] = g.a.values[idx] - lo
        R[idx] = hi - g.b.values[idx]
    return L, R

parts = []
gid = 0
for (sess, line, gain), g in man.groupby(["session", "line", "gain"], sort=True):
    g = g.sort_values("a").reset_index(drop=True)
    comp = components(g)
    g["comp"] = comp
    g["group_id"] = gid + comp
    gid += comp.max() + 1
    g["dL"], g["dR"] = direct_ext(g)
    g["cL"], g["cR"] = chain_ext(g, comp)
    parts.append(g)
d = pd.concat(parts, ignore_index=True)
d["group_size"] = d.groupby("group_id").crop_key.transform("size")
# dedupe: one crop per (group, file) already since one gain per group
keep = ["file_name", "crop_key", "session", "line", "gain", "a", "b", "primary_class", "subset", "fold", "group_id", "group_size", "dL", "dR", "cL", "cR"]
d[keep].to_csv(OUT / "crops.csv", index=False)

# group-level
gg = d.groupby("group_id").agg(session=("session", "first"), line=("line", "first"), gain=("gain", "first"),
                               n=("crop_key", "size"), lo=("a", "min"), hi=("b", "max"),
                               subset=("subset", lambda s: "|".join(sorted(set(s)))),
                               fold=("fold", lambda s: "|".join(sorted(set(s.astype(str))))),
                               classes=("primary_class", lambda s: "|".join(sorted(set(s)))),
                               ).reset_index()
gg["span"] = gg.hi - gg.lo
gg.to_csv(OUT / "groups.csv", index=False)

# adjacent overlapping/touching pairs (consecutive in sorted order inside group)
pairs = []
for gid_, g in d[d.group_size > 1].sort_values(["group_id", "a"]).groupby("group_id"):
    ks = g.reset_index(drop=True)
    for i in range(len(ks) - 1):
        pairs.append((gid_, ks.primary_class[i], ks.primary_class[i + 1], ks.subset[i], ks.b[i] - ks.a[i + 1], ks.a[i + 1] - ks.a[i]))
pp = pd.DataFrame(pairs, columns=["group_id", "c1", "c2", "subset", "overlap", "stride"])
pp.to_csv(OUT / "adjacent_pairs.csv", index=False)

for name, sub in (("trainpool", d[d.subset == "trainpool"]), ("test", d[d.subset == "test"]), ("all", d)):
    print(f"\n=== {name}: crops={len(sub)} groups={sub.group_id.nunique()} ===")
    rows = []
    for cls, s in sub.groupby("primary_class"):
        n = len(s)
        grp = s.groupby("group_id").size()
        multi = grp[grp > 1]
        pr = pp[pp.group_id.isin(s.group_id) & (pp.c1 == cls)]
        row = dict(cls=cls, crops=n, groups=len(grp), multi_groups=len(multi), crops_in_multi=int(multi.sum()),
                   adj_pairs=len(pr), max_group=int(grp.max()))
        for lab, L, R in (("direct", "dL", "dR"), ("chain", "cL", "cR")):
            ext = np.maximum(s[L], s[R])
            both = np.minimum(s[L], s[R])
            row[f"{lab}_any"] = round((ext > 0).mean() * 100, 1)
            for t in (32, 64, 96):
                row[f"{lab}_ge{t}"] = round((ext >= t).mean() * 100, 1)
            row[f"{lab}_both64"] = round((both >= 64).mean() * 100, 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

# leakage: groups across fold / subset
print("\nGROUP leak checks")
print("groups spanning >1 subset:", int(gg.subset.str.contains(r"\|").sum()))
tp = gg[gg.subset == "trainpool"]
print("trainpool groups spanning >1 fold:", int(tp.fold.str.contains(r"\|").sum()))
sess_fold = d[d.subset == "trainpool"].groupby("session").fold.nunique()
print("trainpool sessions with >1 fold:", int((sess_fold > 1).sum()), "of", len(sess_fold))
sess_sub = d.groupby("session").subset.nunique()
print("sessions in both test and trainpool:", int((sess_sub > 1).sum()))
# gain-stripped crop key vs groups: crop_key sets across folds
ck = d.groupby("crop_key").agg(nf=("fold", lambda s: s.astype(str).nunique()), ns=("subset", "nunique"))
print("crop_keys spanning >1 fold/subset:", int(((ck.nf > 1) | (ck.ns > 1)).sum()))
# line key across folds/test
ln = d.groupby("line").agg(nf=("fold", lambda s: s.astype(str).nunique()), ns=("subset", "nunique"))
print("lines spanning >1 fold/subset:", int(((ln.nf > 1) | (ln.ns > 1)).sum()))
# cross-gain: crops with overlap on same line ignoring gain (different x offsets by gain)
x_gain = d.groupby(["line"]).gain.nunique()
print("lines with >1 gain:", int((x_gain > 1).sum()), "of", len(x_gain))
# pair size dist
print("group size distribution (multi):", d[d.group_size > 1].groupby("group_id").size().value_counts().sort_index().to_dict())
print("stride quantiles (adjacent pairs):", pp.stride.quantile([0, .1, .5, .9, 1]).tolist() if len(pp) else None)
print("overlap quantiles:", pp.overlap.quantile([0, .1, .5, .9, 1]).tolist() if len(pp) else None)
print("touching-only pairs (overlap==0):", int((pp.overlap == 0).sum()))
