"""Extra checks: per-activity dup coverage + trace-header timestamps, 09.5 text header diff,
and non-blank aligned-run partial overlap search (beyond whole-file duplicates).

Reads and writes data/analysis/fourtu_dup/ (git-ignored): fourtu_dup_scan.json and
fourtu_dup_pairs.json in, fourtu_dup_partial.json out; the log is stdout.
Run order: fourtu_dup_scan -> fourtu_dup_analyze -> fourtu_dup_extra ->
fourtu_dup_build_csv. Example (from the project root, PowerShell):
    $log = "data/analysis/fourtu_dup/fourtu_dup_extra.log"
    conda run -n gpr-dev python -I analysis/data/fourtu_dup_extra.py *> $log
"""
import json
import difflib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = load_dataset_paths().fourtu
A = PROJECT_ROOT / "data" / "analysis" / "fourtu_dup"
A.mkdir(parents=True, exist_ok=True)
D = json.loads((A / "fourtu_dup_scan.json").read_text(encoding="utf-8"))
F = D["files"]
for r in F:
    r["key"] = f"{r['activity']}/{r['path']}"
byk = {r["key"]: r for r in F}
pairs = json.loads((A / "fourtu_dup_pairs.json").read_text())["pairs"]
dupkeys = {k for p in pairs for k in p}

def tstr(r):
    y, d, h, m, s = r["sgy_time_le"]
    return f"{y}-d{d:03d} {h:02d}:{m:02d}:{s:02d}"

acts = sorted({k.split("/")[0] for k in dupkeys})
print("== per-activity coverage and times")
for a in acts:
    paths = [r for r in F if r["activity"] == a]
    nd = sum(1 for r in paths if r["key"] in dupkeys)
    ts = sorted(tstr(r) for r in paths)
    print(a, "paths", len(paths), "dup", nd, "time range", ts[0], "..", ts[-1])
    nondup = [(r["path"], tstr(r)) for r in paths if r["key"] not in dupkeys]
    if nondup:
        print("   non-dup paths:", nondup[:6])

print("== time equality across pairs (trace-header, first trace)")
bad = 0
for a, b in pairs:
    if byk[a]["sgy_time_le"] != byk[b]["sgy_time_le"]:
        bad += 1
print("pairs with different first-trace time:", bad)

print("== 09.5 text header diff Path8 vs Path13")
ta = byk["09.5/Path8"]["sgy_text_ascii"]
tb = byk["09.5/Path13"]["sgy_text_ascii"]
def lines(t): return [t[i:i + 80] for i in range(0, len(t), 80)]
for l in difflib.unified_diff(lines(ta), lines(tb), lineterm="", n=0):
    print("  ", l.rstrip().encode("ascii", "replace").decode())

print("== 09.5 within-activity pair times / trace counts")
for a, b in pairs:
    if a.startswith("09.5/"):
        print(a, b, tstr(byk[a]), tstr(byk[b]), byk[a]["shape"][1], byk[a]["zip_dt"], byk[b]["zip_dt"], byk[a]["source_member"], byk[b]["source_member"])

print("== text-header-only comparison among all dups (activity strings)")
for a, b in pairs[:3]:
    print(a, b, byk[a]["sgy_text_ascii"][:160].replace("\n", " ").encode("ascii", "replace").decode())

# partial overlap: non-blank aligned runs
print("== non-blank aligned-run partial overlaps (not whole-file dups)")
cols = {}
for r in F:
    arr = np.load(ROOT / "05_radargrams_npy" / r["activity"] / f"{r['path']}.npy")
    nonblank = arr.std(axis=0) > 3.0
    hs = [hash(arr[:, i].tobytes()) if nonblank[i] else None for i in range(arr.shape[1])]
    cols[r["key"]] = hs
idx = defaultdict(list)
for k, hs in cols.items():
    for i, h in enumerate(hs):
        if h is not None:
            idx[h].append((k, i))
cand = defaultdict(lambda: defaultdict(int))   # (a,b) -> offset -> count
for h, lst in idx.items():
    if 1 < len(lst) <= 4:
        for (ka, ia), (kb, ib) in combinations(lst, 2):
            if ka == kb:
                continue
            if ka > kb:
                ka, kb, ia, ib = kb, ka, ib, ia
            cand[(ka, kb)][ia - ib] += 1
pairset = {tuple(sorted(p)) for p in pairs}
out = []
for (ka, kb), offs in cand.items():
    if (ka, kb) in pairset:
        continue
    off, n = max(offs.items(), key=lambda x: x[1])
    if n >= 10:
        out.append((ka, kb, off, n, len(cols[ka]), len(cols[kb])))
out.sort(key=lambda x: -x[3])
print("pairs with >=10 non-blank traces matching at one fixed offset:", len(out))
for o in out[:60]:
    print("  ", o)
(A / "fourtu_dup_partial.json").write_text(json.dumps(out), encoding="utf-8")
