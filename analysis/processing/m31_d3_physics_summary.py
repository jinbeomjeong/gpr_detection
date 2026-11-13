"""Physics-side summary of the D3 4TU qualitative run (spec_detection_image_v1 §13).

Reads data/processing/d3/*.csv only (no depth, no 4TU ground truth). Prints:
  - R2 duplicate pairs: top difference in periods (V1 threshold 1 P), fold 0 and all folds
  - band neighbour pairs: top difference (px, P), smaller height, time overlap, why not candidate
  - line flags: overlap_period_capped / compression distribution, P_line by dt
  - known apexes missed by fold 0: nearest pipe_hyperbola / any box and the miss geometry
Writes analysis/processing/m31_d3_physics_summary.json.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

D3 = Path(r"C:\workspace\gpr_detection\data\processing\d3")
OUT = Path(r"C:\workspace\gpr_detection\analysis\processing\m31_d3_physics_summary.json")


def rows(name):
    with open(D3 / f"{name}.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def q(vals, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(vals, float)
    if a.size == 0:
        return {"n": 0}
    d = {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps}
    d.update(n=int(a.size), max=round(float(a.max()), 3))
    return d


def tb(v):
    return v == "True"


out = {}

r2 = rows("r2_pairs")
for scope, sel in (("fold0", [r for r in r2 if r["fold"] == "0"]), ("all", r2)):
    td = [float(r["top_diff_p"]) for r in sel]
    out[f"r2_{scope}"] = {
        "top_diff_p": q(td),
        "n_ge_0p5": sum(t >= 0.5 for t in td), "n_ge_1p0": sum(t > 1.0 for t in td),
        "gap_over_limit": q([float(r["gap_over_limit"]) for r in sel]),
        "either_truncated": sum(tb(r["either_truncated"]) for r in sel),
        "v1": sum(tb(r["v1_veto"]) for r in sel), "v2": sum(tb(r["v2_veto"]) for r in sel),
    }

bp = rows("band_pairs")
for scope, sel in (("fold0", [r for r in bp if r["fold"] == "0"]), ("all", bp)):
    cand = [r for r in sel if tb(r["is_candidate"])]
    notc = [r for r in sel if not tb(r["is_candidate"])]
    reasons = Counter()
    for r in notc:
        low = float(r["time_overlap_frac"]) < 0.5
        reasons["overlap<0.5" if low else "gap_only"] += 1
    out[f"band_{scope}"] = {
        "pairs": len(sel), "not_candidate": len(notc), "reasons": dict(reasons),
        "top_diff_px_all": q([float(r["top_diff_px"]) for r in sel]),
        "top_diff_p_all": q([float(r["top_diff_p"]) for r in sel]),
        "smaller_height_px": q([float(r["smaller_height_px"]) for r in sel]),
        "top_diff_px_not_candidate": q([float(r["top_diff_px"]) for r in notc]),
        "height_px_not_candidate": q([float(r["smaller_height_px"]) for r in notc]),
        "ratio_topdiff_over_height_not_candidate": q(
            [float(r["top_diff_px"]) / max(float(r["smaller_height_px"]), 1e-9) for r in notc]),
        "lines_with_band_pairs": len({r["line"] for r in sel}),
    }

lf = rows("line_flags")
out["flags"] = {
    "lines": len(lf),
    "capped_lines": [(r["line"], r["eps_r_used"], r["period_samples"]) for r in lf if tb(r["overlap_period_capped"])],
    "compressed": sum(tb(r["period_compressed"]) for r in lf),
    "scale_out_of_range": sum(tb(r["scale_out_of_range"]) for r in lf),
    "period_source": dict(Counter(r["period_source"] for r in lf)),
    "content_height_px": q([float(r["content_height_px"]) for r in lf]),
    "overlap_phys_px": q([float(r["overlap_phys_px"]) for r in lf]),
}
by_period = defaultdict(list)
for r in lf:
    by_period[r["eps_r_used"]].append(float(r["period_samples"]))

# apex misses
boxes = [b for b in rows("boxes") if b["fold"] == "0"]
by_line = defaultdict(list)
for b in boxes:
    by_line[b["line"]].append(b)
period = {r["line"]: float(r["period_samples"]) for r in lf}
cases = []
for a in rows("known_apexes"):
    line, tr, smp = a["line"], int(a["trace"]), float(a["sample"])
    p = period[line]
    hyp = [b for b in by_line[line] if b["class_name"] == "pipe_hyperbola"]
    other = [b for b in by_line[line] if b["class_name"] != "pipe_hyperbola"]

    def geom(b):
        i0, i1, k0, k1 = (int(b[k]) for k in ("i0", "i1", "k0", "k1"))
        dtr = 0 if i0 <= tr <= i1 else min(abs(tr - i0), abs(tr - i1))
        if smp < k0:
            dk = (smp - k0) / p  # negative: apex above box top
        elif smp > k1:
            dk = (smp - k1) / p  # positive: apex below box bottom
        else:
            dk = 0.0
        return dtr, dk

    rec = {"id": a["blind_id"], "line": line, "hit": tb(a["hit_fold0"]), "folds_hit": int(a["folds_hit"])}
    if hyp:
        g = sorted((geom(b) + (b,) for b in hyp), key=lambda x: (x[0], abs(x[1])))[0]
        rec.update(nearest_hyp_dtrace=g[0], nearest_hyp_dk_periods=round(g[1], 2),
                   nearest_hyp_score=float(g[2]["score"]))
    else:
        rec.update(nearest_hyp_dtrace=None)
    ov = [b for b in other if geom(b)[0] == 0 and geom(b)[1] == 0.0]
    rec["other_class_covering"] = sorted({b["class_name"] for b in ov})
    cases.append(rec)

miss = [c for c in cases if not c["hit"]]
cause = Counter()
for c in miss:
    if c["nearest_hyp_dtrace"] is None:
        k = "no_hyperbola_box_on_line"
    elif c["nearest_hyp_dtrace"] > 3:
        k = "no_box_at_apex_trace(>3 tr)"
    elif c["nearest_hyp_dk_periods"] < -0.5:
        k = "apex_above_box_top(>0.5P)"
    elif c["nearest_hyp_dk_periods"] > 0.5:
        k = "apex_below_box_bottom(>0.5P)"
    else:
        k = "within_tolerance?"
    if c["other_class_covering"]:
        k += "+covered_by_" + "/".join(c["other_class_covering"])
    cause[k] += 1
out["apex"] = {"n": len(cases), "hit_fold0": sum(c["hit"] for c in cases), "miss_causes": dict(cause),
               "miss_cases": miss,
               "above_top_dk_periods": q([c["nearest_hyp_dk_periods"] for c in miss
                                          if c.get("nearest_hyp_dk_periods") is not None and c["nearest_hyp_dtrace"] <= 3
                                          and c["nearest_hyp_dk_periods"] < 0])}
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "apex"}, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out["apex"].items() if k != "miss_cases"}, indent=1, ensure_ascii=False))
for c in miss:
    print(c)
