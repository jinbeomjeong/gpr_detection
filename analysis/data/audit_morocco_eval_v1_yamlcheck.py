"""Ultralytics check_det_dataset on morocco_eval_v1/eval.yaml (separate process: no matplotlib)."""
from ultralytics.data.utils import check_det_dataset

d = check_det_dataset(r"C:\workspace\gpr_detection\data\morocco_eval_v1\eval.yaml", autodownload=False)
print("nc", d["nc"], d["names"])
for k in ("train", "val", "test"):
    print(k, d[k])
lines = open(d["val"], encoding="utf-8").read().splitlines()
print("val lines", len(lines))
