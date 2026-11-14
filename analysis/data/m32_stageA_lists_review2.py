"""Second read-only check: K bundle lines vs known_apexes.csv; IDS eligible zone/mission shares; AIHub fold balance in pool."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\workspace\gpr_detection")
GZ = next(Path(r"C:\workspace\GPR_Data").glob("광저우대_*"))


def rows(p):
    with open(p, encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


ka = rows(ROOT / "data" / "processing" / "d3" / "known_apexes.csv")
print("known_apexes cols", list(ka[0]))
d3 = rows(ROOT / "data" / "processing" / "d3" / "lines.csv")
kset = {f"{r['activity_id']}/{r['path']}" for r in d3 if "K" in r["bundles"]}
kcols = list(ka[0])
act = "line"
pth = None
ka_lines = {r["line"] for r in ka}
print("n apex rows", len(ka), "apex lines", len(ka_lines), "K lines", len(kset))
print("K - apex", sorted(kset - ka_lines), "apex - K", sorted(ka_lines - kset))
print("01.1 apexes", sum(1 for r in ka if r["line"].startswith("01.1/")))
idx = rows(GZ / "01_메타데이터" / "radargrams_index.csv")
el = [r for r in idx if r["target"] == "pipe" and r["format"] != "MALA .rd3"]
print("eligible mission", Counter(r["mission"] for r in el))
z = Counter(r["zone"] for r in el)
print("eligible zones", len(z), z.most_common(6))
li = [r["line"] for r in rows(ROOT / "analysis" / "processing" / "m32_stageA_lines_ids.csv")]
m = {r["file_id"]: r for r in idx}
print("list zone 190118AA share of eligible", z["190118AA"], "list 17")
print("eligible n_traces/len median", sorted(float(r["length_m"]) for r in el)[len(el) // 2])
