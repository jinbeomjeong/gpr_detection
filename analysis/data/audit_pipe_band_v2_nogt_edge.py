"""no_gt FP in same depth row: how many touch the crop edge / lie in the tail beyond the last GT box."""
from pathlib import Path

import pandas as pd

OUT = Path(r"C:\workspace\gpr_detection\analysis\data\out")
ng = pd.read_csv(OUT / "pipe_band_v2_nogt_fp.csv")
sr = ng[ng.same_row].copy()
sr["touch_l"] = sr.px1 <= 3
sr["touch_r"] = sr.px2 >= 253
sr["touch"] = sr.touch_l | sr.touch_r
sr["is_tail"] = ((sr.px2 <= sr.gx1) & sr.touch_l) | ((sr.px1 >= sr.gx2) & sr.touch_r)  # FP between GT box and the crop edge
print("same-row no_gt FP", len(sr), "touch crop edge", int(sr.touch.sum()), "tail beyond GT (edge side)", int(sr["is_tail"].sum()))
print("not touching edge:", sr[~sr.touch][["crop_key", "gap_to_gt", "yov"]].round(1).to_string(index=False))
print("tail gap quantiles", sr[sr["is_tail"]].gap_to_gt.quantile([.25, .5, .75, 1]).round(1).tolist())
print("FP width median", (sr.px2 - sr.px1).median(), " touch FP width median", (sr[sr.touch].px2 - sr[sr.touch].px1).median())
allng = ng.copy()
allng["touch"] = (allng.px1 <= 3) | (allng.px2 >= 253)
print("all no_gt FP touching edge:", int(allng.touch.sum()), "/", len(allng), " same_row:", int(allng[allng.same_row].touch.sum()), " not same_row:", int(allng[~allng.same_row].touch.sum()), "/", int((~allng.same_row).sum()))
# neighbouring-crop view: is the tail a GT box in the overlapping neighbour crop? (needs crop overlap; reported as count of crops only)
print(sr.crop_key.nunique(), "crops")

