"""Cause decomposition of the M2-8 4TU result (diagnostic only, no setting is changed).

Reads the committed outputs (depth_4tu_v1_candidates.csv, fourtu_apex_candidates_v1.csv,
fourtu_lines_v1.csv) and the 4TU ground truth, and reuses the matching / baseline
functions of scripts/eval_depth_4tu.py read-only. Main combination only
(top assumption, detected zero, gt set exclude_x, support >= 2 lines).

1. candidate_moved: trace / time / depth shift between the apex candidate and the
   apex that ``estimate_depth`` re-seeded inside the supplied box.
2. Single-linkage chaining: cluster span and size per activity.
3. Factor swap: {re-seeded depths, candidate-direct depths v_meta * tau_cand / 2}
   x {single-linkage chain (as run), span-limited clusters (max span 0.10 m)}.
   Each with the activity-range baseline (200 trials, default_rng(0)).
4. Ground-truth presence per site; depth distributions of estimates and GT.

Run:  conda run -n gpr-dev python -I analysis/processing/depth_4tu_v1_diagnose.py
Out:  docs/processing/depth_4tu_v1_diagnose.csv (factor table), stdout summaries.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_depth_4tu import (  # noqa: E402
    cluster_depths,
    load_ground_truth_sets,
    match_activity,
    random_baseline,
    summarize,
    supported,
)
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOC = ROOT / "docs" / "processing"
TRIALS = 200


def span_limited(depths, lines, span_m=0.10):
    """Greedy clusters whose max - min <= span_m (no chaining)."""
    from eval_depth_4tu import Cluster

    order = np.argsort(depths)
    d = np.asarray(depths, float)[order]
    ln = [lines[i] for i in order]
    out, start = [], 0
    for i in range(1, len(d) + 1):
        if i == len(d) or d[i] - d[start] > span_m:
            out.append(Cluster(depth_m=float(np.median(d[start:i])), n_lines=len(set(ln[start:i])),
                               n_apex_candidates=i - start))
            start = i
    return out


def evaluate(est_by_act, lines_by_act, n_lines_act, gt, clusterer):
    matches, n_est = {}, {}
    for act, gts in gt.items():
        depths = est_by_act.get(act, [])
        clusters = clusterer(depths, lines_by_act.get(act, [])) if depths else []
        kept = supported(clusters, n_lines_act.get(act, 1), 2)
        est = [c.depth_m for c in kept]
        matches[act] = match_activity(est, gts)
        n_est[act] = len(est)
    s = summarize(matches, {a: len(g) for a, g in gt.items()})
    b = random_baseline({a: n for a, n in n_est.items()}, gt, trials=TRIALS, seed=0)
    return s, b


def main() -> None:
    cand = pd.read_csv(DOC / "depth_4tu_v1_candidates.csv", dtype={"activity_id": str})
    seeds = pd.read_csv(DOC / "fourtu_apex_candidates_v1.csv", dtype={"activity_id": str})
    lines = pd.read_csv(DOC / "fourtu_lines_v1.csv", dtype={"activity_id": str})
    dt = dict(zip(zip(lines["activity_id"], lines["path"]), lines["dt_ns"], strict=True))
    n_lines_act = lines.groupby("activity_id").size().to_dict()

    # 1. candidate_moved
    m = cand.merge(seeds[["activity_id", "path", "trace", "sample", "tau_ns"]],
                   left_on=["activity_id", "path", "candidate_trace", "candidate_sample"],
                   right_on=["activity_id", "path", "trace", "sample"], how="left")
    m["dt"] = [dt[(a, p)] for a, p in zip(m["activity_id"], m["path"], strict=True)]
    ok = m[m["status"] == "ok"].copy()
    ok["d_trace"] = ok["apex_trace"] - ok["candidate_trace"]
    ok["d_ns"] = (ok["apex_sample"] - ok["candidate_sample"]) * ok["dt"]
    ok["v"] = [velocity_from_eps(e) for e in ok["eps_r_used"]]
    ok["depth_cand"] = ok["v"] * ok["tau_ns"] / 2.0
    ok["d_depth"] = ok["depth_top_m"] - ok["depth_cand"]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    print("[1] |d_trace| quantiles", ok["d_trace"].abs().quantile(q).round(1).tolist())
    print("    d_ns quantiles", ok["d_ns"].quantile(q).round(2).tolist(),
          " earlier share", round(float((ok["d_ns"] < -0.5).mean()), 3),
          " later share", round(float((ok["d_ns"] > 0.5).mean()), 3))
    print("    depth(reseeded) - depth(candidate) m quantiles", ok["d_depth"].quantile(q).round(3).tolist())

    # ground truth
    gt_sets = load_ground_truth_sets(BASE)
    gt_items = gt_sets["exclude_x"] if "exclude_x" in gt_sets else next(iter(gt_sets.values()))
    gt = {a: [g.depth_m for g in items] for a, items in gt_items.items() if items}
    site_of = lines.groupby("activity_id")["site"].first().to_dict()
    gt_sites = defaultdict(int)
    for a, g in gt.items():
        gt_sites[site_of.get(a)] += len(g)
    print("[4] GT rows per site:", dict(sorted(gt_sites.items())))
    all_gt = np.concatenate([np.asarray(v) for v in gt.values()])
    print("    GT depth quantiles", np.quantile(all_gt, q).round(2).tolist())
    print("    re-seeded depth quantiles", ok["depth_top_m"].quantile(q).round(2).tolist(),
          " candidate-direct", ok["depth_cand"].quantile(q).round(2).tolist())

    # 2. chaining
    rs = defaultdict(list)
    ln = defaultdict(list)
    cd = defaultdict(list)
    for r in ok.itertuples():
        rs[r.activity_id].append(r.depth_top_m)
        cd[r.activity_id].append(r.depth_cand)
        ln[r.activity_id].append(r.path)
    spans, sizes = [], []
    for a in rs:
        for c in cluster_depths(rs[a], ln[a], 0.10):
            members = sorted(rs[a])
            sizes.append(c.n_apex_candidates)
        cl = cluster_depths(rs[a], ln[a], 0.10)
        srt = np.sort(rs[a])
        start = 0
        for c in cl:
            seg = srt[start:start + c.n_apex_candidates]
            spans.append(float(seg.max() - seg.min()))
            start += c.n_apex_candidates
    print("[2] chained cluster span m quantiles", np.quantile(spans, q).round(2).tolist(),
          " size quantiles", np.quantile(sizes, q).round(1).tolist(),
          " share of estimates in clusters spanning > 0.3 m",
          round(sum(s for s, sp in zip(sizes, spans, strict=True) if sp > 0.3) / sum(sizes), 3))

    # 3. factor swap
    chain = lambda d, l: cluster_depths(d, l, 0.10)  # noqa: E731
    rows = []
    for est_name, est in (("reseeded (as run)", rs), ("candidate_direct", cd)):
        for cl_name, cl in (("single_linkage (as run)", chain), ("span_limited_0.10", span_limited)):
            s, b = evaluate(est, ln, n_lines_act, gt, cl)
            row = {"depths": est_name, "clusters": cl_name, "n_gt": s.n_gt, "n_est": s.n_est,
                   "n_pairs": s.n_pairs, "mae_m": round(s.mae_m, 3), "bias_m": round(s.bias_m, 3),
                   "pass_a": round(s.pass_a, 3), "pass_b": round(s.pass_b, 3), "missed": s.missed,
                   "false_est": s.false_est, "baseline_pass_b_mean": round(b.pass_b_mean, 3),
                   "baseline_pass_b_p975": round(b.pass_b_p975, 3),
                   "baseline_p": round(float(np.mean(b.pass_b_values >= s.pass_b)), 3)}
            rows.append(row)
            print("[3]", row, flush=True)
    with open(DOC / "depth_4tu_v1_diagnose.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
