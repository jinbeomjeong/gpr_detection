"""Print fold x class table (images/boxes/sessions) from summary.json and balance info."""
import json
from pathlib import Path

s = json.loads(Path(r"C:\workspace\gpr_detection\data\aihub_yolo_v1\summary.json").read_text(encoding="utf-8"))
for f in s["folds"]:
    row = [f["fold"], f["images"], f["boxes"], f["crops"], f["sessions"]]
    for c, v in f["classes"].items():
        row.append(f"{c}:{v['images']}/{v['boxes']}/{v['sessions']}")
    print(row)
t = s["subsets"]["test"]
print("test", t["images"], t["boxes"], t["crops"], t["sessions"], {c: (v["images"], v["boxes"], v["sessions"]) for c, v in t["classes"].items()})
tp = s["subsets"]["trainpool"]
print("pool", tp["images"], tp["boxes"], tp["crops"], tp["sessions"], {c: (v["images"], v["boxes"], v["sessions"]) for c, v in tp["classes"].items()})
print("balance", json.dumps(s["balance"], ensure_ascii=False))
print("test_selection", json.dumps(s["test_selection"], ensure_ascii=False))
print("source", json.dumps(s["source"]))
print("selected", s["selected"]["images"], s["selected"]["boxes"])
