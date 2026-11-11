"""M2-10 (d) diagnosis of wrong-event selection in the 4TU shallow tail.

ALREADY-SEEN DATA, EXPLORATION ONLY. 4TU results were seen in the M2-9
re-evaluation (commit 1d50fbc); every number written here is a diagnostic
count, not a verdict, and must not be used to tune a rule for 4TU.

Inputs (read only):
* docs/processing/depth_4tu_v2_candidates.csv, _clusters.csv, _matches.csv
  (main run, commit db5c658)
* docs/processing/fourtu_apex_candidates_v1.csv, fourtu_lines_v1.csv
* data/processing/m210d_generator_items.csv, m210d_generator_lines.csv
  (from m210d_generator_stages.py: generator peaks / accepted / kept)
* docs/processing/m29c_apex_review.csv (manual review labels, blind to truth)
* 4TU ground-truth tables (01_메타데이터), parsed with the evaluation code

Steps:
1. Rebuild the main estimates (B+D+E QC, complete linkage 0.10 m, support
   >= 2 lines) and the ungated Hungarian with the evaluation functions and check
   that they reproduce the committed clusters / matches (184 pairs, 140 pass).
2. For every truth item (exclude_x, pipe top, detected zero) count the items
   within +-0.30 m of its depth at each pipeline stage:
   peak -> accepted -> kept (generator) -> fit ok -> not B -> not D -> not E
   -> supported cluster -> final pair. The first stage with no item is the
   loss stage. Chance coverage of the same +-0.30 m window is reported for
   each stage (fraction of a depth grid covered), because a near item can be
   a coincidence.
3. Classify the 44 failed pairs and 204 missed truths by mechanism
   (i) lost before the estimates (by stage), (ii) assignment conflict,
   (iii) the paired shallow estimate (second event of a listed pipe /
   unlisted event / near-surface), (iv) truth-side flags.
4. Label-free stage survival versus candidate depth (no truth used).

Outputs:
* docs/processing/m210d_truth_classes.csv  (one row per truth item)
* docs/processing/m210d_failed_pairs.csv   (one row per failed pair)
* docs/processing/m210d_zero_activities.csv (18 activities without estimate)
* docs/processing/m210d_stage_depth_labelfree.csv
* docs/processing/m210d_summary.json
* data/processing/m210d_estimates.csv (cluster members, git-ignored)

Run: conda run -n gpr-dev python -I analysis/processing/m210d_event_selection.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10,
# same setting as scripts/eval_depth_4tu.py); an explicit value wins.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_depth_4tu as ev  # noqa: E402  (read-only import of the evaluation code)

from gpr_detection.io.fourtu import FourTUTable, load_fourtu_table  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = ROOT / "docs" / "processing"
DATA = ROOT / "data" / "processing"
ID = {"activity_id": str, "path": str, "line_id": str}
LIMIT = ev.PASS_LIMIT_M + ev.PASS_TOL_M
RECORD_END_NS = 5.0  # D rule: 2.5 periods at 500 MHz (spec_eval 3-5)
NEAR_SURFACE_M = 0.25  # assumption: paving / sub-base layer depth for labelling only
GRID = np.round(np.arange(0.15, 2.505, 0.01), 3)

STAGES = (
    "peak",
    "accepted",
    "kept",
    "fit_ok",
    "not_B",
    "not_D",
    "not_E",
    "supported",
)
LOSS_LABEL = {
    "peak": "i-gen0 no peak (no detectable energy)",
    "accepted": "i-gen1 rejected by acceptance rule",
    "kept": "i-gen2 cut by ranking/NMS/cap 6",
    "fit_ok": "i-fit depth fit failed",
    "not_B": "i-qcB curvature out",
    "not_D": "i-qcD record end / seam",
    "not_E": "i-qcE curvature unknown",
    "supported": "i-sup support filter / cluster",
    "final": "ii assignment conflict",
}


def read_inputs():
    cand = pd.read_csv(DOCS / "depth_4tu_v2_candidates.csv", dtype=ID, keep_default_na=False)
    clusters = pd.read_csv(DOCS / "depth_4tu_v2_clusters.csv", dtype=ID)
    matches = pd.read_csv(DOCS / "depth_4tu_v2_matches.csv", dtype=ID)
    lines = pd.read_csv(DOCS / "fourtu_lines_v1.csv", dtype=ID)
    gen_items = pd.read_csv(DATA / "m210d_generator_items.csv", dtype=ID)
    gen_lines = pd.read_csv(DATA / "m210d_generator_lines.csv", dtype=ID)
    review = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype=ID)
    dups = pd.read_csv(ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv", dtype=str, encoding="utf-8-sig")
    gt_table = load_fourtu_table(BASE, FourTUTable.GROUNDTRUTH)
    meta = load_fourtu_table(BASE, FourTUTable.UTILITIES_METADATA)
    return cand, clusters, matches, lines, gen_items, gen_lines, review, dups, gt_table, meta


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower() == "true"


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def main() -> None:
    cand, clusters_csv, matches_csv, lines, gen_items, gen_lines, review, dups, gt_table, meta = read_inputs()
    within = dups[dups["scope"] == "within_activity"]
    copies = set(zip(within["activity_b"], within["path_b"]))
    cand["is_copy"] = [(a, p) in copies for a, p in zip(cand["activity_id"], cand["path"])]
    cand = cand[~cand["is_copy"]].copy()
    cand["depth"] = to_float(cand["depth_top_m"])
    for col in ("qc_excluded", "apex_at_record_end", "apex_on_trace_seam"):
        cand[col] = as_bool(cand[col])
    cand["candidate_trace"] = cand["candidate_trace"].astype(int)
    cand["candidate_sample"] = to_float(cand["candidate_sample"])
    cand["tau"] = np.nan
    v1 = pd.read_csv(DOCS / "fourtu_apex_candidates_v1.csv", dtype=ID)
    tau_map = {
        (a, p, int(t), round(float(s), 2)): tau
        for a, p, t, s, tau in zip(v1["activity_id"], v1["path"], v1["trace"], v1["sample"], v1["tau_ns"])
    }
    cand["tau"] = [
        tau_map.get((a, p, t, round(s, 2)), np.nan)
        for a, p, t, s in zip(cand["activity_id"], cand["path"], cand["candidate_trace"], cand["candidate_sample"])
    ]
    vmeta = {(a, p): v for a, p, v in zip(gen_lines["activity_id"], gen_lines["path"], gen_lines["v_meta"])}
    cand["depth_nom"] = [
        vmeta.get((a, p), np.nan) * t / 2 if np.isfinite(t) else np.nan
        for a, p, t in zip(cand["activity_id"], cand["path"], cand["tau"])
    ]
    cand["row"] = np.arange(len(cand))

    # ------------------------------------------------------------ truth
    gt_sets = ev.build_ground_truth_sets(gt_table, meta)
    gt = gt_sets["exclude_x"]
    gt_rows = gt_table[gt_table["depth_m"].str.strip() != ""]
    gt_rows = gt_rows[gt_rows["code"] != ev.CODE_NOT_A_PIPE].reset_index(drop=True)

    # ------------------------------------------------------------ step 1: rebuild main
    line_counts = Counter(
        a for a, p in zip(lines["activity_id"], lines["path"]) if (a, p) not in copies
    )
    keep = (cand["status"] == "ok") & (~cand["qc_excluded"]) & cand["depth"].notna()
    est_rows = []
    estimates: dict[str, list[float]] = {}
    for act in sorted(line_counts):
        sub = cand[keep & (cand["activity_id"] == act)]
        cl = ev.cluster_depths(list(sub["depth"]), list(sub["path"]), ev.CLUSTER_WIDTH_M, "complete")
        kept = ev.supported(cl, line_counts[act], ev.MIN_SUPPORT_LINES)
        estimates[act] = [c.depth_m for c in kept]
        for ci, c in enumerate(cl):
            members = sub.iloc[list(c.member_indices)]
            est_rows.append(
                {
                    "activity_id": act,
                    "cluster_index": ci,
                    "depth_m": c.depth_m,
                    "n_lines": c.n_lines,
                    "n_members": c.n_apex_candidates,
                    "kept_by_support": c in kept,
                    "est_index": kept.index(c) if c in kept else None,
                    "member_rows": " ".join(str(r) for r in members["row"]),
                    "member_paths": " ".join(members["path"]),
                    "member_traces": " ".join(str(t) for t in members["candidate_trace"]),
                    "member_sources": " ".join(members["source"]),
                    "member_qc": " ".join(members["qc_verdict"]),
                }
            )
    est_df = pd.DataFrame(est_rows)
    # check against committed clusters csv
    ref = clusters_csv[clusters_csv["zero_variant"] == "detected"]
    ok_clusters = len(ref) == len(est_df) and np.allclose(
        np.sort(ref["depth_m"].to_numpy()), np.sort(est_df["depth_m"].round(4).to_numpy()), atol=2e-4
    )
    n_kept_ref = int(as_bool(ref["kept_by_support"]).sum())
    n_kept_mine = int(est_df["kept_by_support"].sum())

    # Hungarian: the committed pairs are authoritative. The L1 cost on a line is
    # highly degenerate (many co-optimal assignments), so a rebuild from the
    # 4-decimal csv depths can pick another optimum; that is reported, not used.
    ref_m = matches_csv[
        (matches_csv["gt_set"] == "exclude_x")
        & (matches_csv["assumption"] == "top")
        & (matches_csv["zero_variant"] == "detected")
        & (matches_csv["variant"] == "main")
    ]
    ref_by_act = defaultdict(list)
    for a, e_, g_ in zip(ref_m["activity_id"], ref_m["est_depth_m"], ref_m["gt_depth_m"]):
        ref_by_act[a].append((float(e_), float(g_)))
    truth_rows = []
    gate_gain = 0
    rebuild_pass = 0
    tie_min = tie_max = 0
    tie_acts = []
    for act, items in gt.items():
        g = [it.depth_for("top") for it in items]
        g_center = [it.depth_for("center") for it in items]
        e = estimates.get(act, [])
        assigned = {}
        if e:
            cost = np.abs(np.subtract.outer(np.asarray(e, float), np.asarray(g, float)))
            rows, cols = linear_sum_assignment(cost)
            passes = cost <= LIMIT
            rebuild_pass += int(passes[rows, cols].sum())
            gm = ev.match_activity_gated(e, g)
            gate_gain += gm.n_pairs - int(passes[rows, cols].sum())
            # range of the pass count over co-optimal assignments (tie-breaking)
            eps = 1e-7
            r_hi, c_hi = linear_sum_assignment(cost - eps * passes)
            r_lo, c_lo = linear_sum_assignment(cost + eps * passes)
            hi, lo = int(passes[r_hi, c_hi].sum()), int(passes[r_lo, c_lo].sum())
            tie_max += hi
            tie_min += lo
            if hi != lo:
                tie_acts.append([act, lo, hi])
            # committed assignment: map (est, gt depth) back to indices
            used_e, used_g = set(), set()
            for e_, g_ in ref_by_act.get(act, []):
                ri = next(i for i, x in enumerate(e) if i not in used_e and abs(round(x, 4) - e_) < 1.5e-4)
                ci = next(j for j, x in enumerate(g) if j not in used_g and abs(round(x, 4) - g_) < 1.5e-4)
                used_e.add(ri)
                used_g.add(ci)
                assigned[ci] = ri
        for gi, it in enumerate(items):
            r = assigned.get(gi)
            truth_rows.append(
                {
                    "activity_id": act,
                    "site": ev.site_of_activity(act),
                    "gt_index": gi,
                    "gt_depth_m": g[gi],
                    "gt_center_shift_m": g[gi] - g_center[gi],
                    "radius_known": it.radius_known,
                    "code": it.code,
                    "multi_trench": it.multi_trench,
                    "n_gt_act": len(items),
                    "n_est_act": len(e),
                    "n_lines_act": line_counts.get(act, 0),
                    "est_index": r,
                    "est_depth_m": None if r is None else e[r],
                    "error_m": None if r is None else e[r] - g[gi],
                    "status": "missed" if r is None else ("pass" if abs(e[r] - g[gi]) <= LIMIT else "failed"),
                }
            )
    truth = pd.DataFrame(truth_rows)
    repro = {
        "rebuild_pass_other_optimum": rebuild_pass,
        "pass_range_over_cooptimal_assignments": [tie_min, tie_max],
        "activities_with_pass_range": tie_acts,
        "clusters_equal_depths": bool(ok_clusters),
        "n_clusters": [int(len(ref)), int(len(est_df))],
        "n_supported": [n_kept_ref, n_kept_mine],
        "n_pairs": [int(len(ref_m)), int(truth["est_index"].notna().sum())],
        "n_pass": [int(as_bool(ref_m["within_30cm"]).sum()), int((truth["status"] == "pass").sum())],
        "n_gt": int(len(truth)),
    }
    # pair-level equality (activity, est, gt)
    mine_pairs = sorted(
        (a, round(float(e), 4), round(float(g), 4))
        for a, e, g in zip(truth["activity_id"], truth["est_depth_m"], truth["gt_depth_m"])
        if e is not None and not pd.isna(e)
    )
    ref_pairs = sorted(zip(ref_m["activity_id"], ref_m["est_depth_m"].round(4), ref_m["gt_depth_m"].round(4)))
    ref_pairs = [(a, float(e), float(g)) for a, e, g in ref_pairs]
    repro["pairs_identical_within_1e-4"] = len(mine_pairs) == len(ref_pairs) and all(
        a == b and abs(e1 - e2) < 1.5e-4 and abs(g1 - g2) < 1.5e-4
        for (a, e1, g1), (b, e2, g2) in zip(mine_pairs, ref_pairs)
    )
    repro["gate_pass_gain_same_estimates"] = int(gate_gain)

    # ------------------------------------------------------------ step 2: stage evidence
    gi_ = gen_items.copy()
    gi_["depth_nom_m"] = to_float(gi_["depth_nom_m"].astype(str))
    stage_items: dict[str, pd.DataFrame] = {
        "peak": gi_[["activity_id", "path", "depth_nom_m"]].rename(columns={"depth_nom_m": "d"}),
        "accepted": gi_[gi_["stage"].isin(["accepted", "kept"])][["activity_id", "path", "depth_nom_m"]].rename(columns={"depth_nom_m": "d"}),
    }
    kept_gen = gi_[gi_["stage"] == "kept"][["activity_id", "path", "depth_nom_m"]].rename(columns={"depth_nom_m": "d"})
    manual = cand[cand["source"] == "manual_v0"][["activity_id", "path", "depth_nom"]].rename(columns={"depth_nom": "d"})
    stage_items["kept"] = pd.concat([kept_gen, manual], ignore_index=True)
    ok = cand[(cand["status"] == "ok") & cand["depth"].notna()]
    stage_items["fit_ok"] = ok[["activity_id", "path", "depth"]].rename(columns={"depth": "d"})
    nb = ok[ok["qc_verdict"] != "out"]
    stage_items["not_B"] = nb[["activity_id", "path", "depth"]].rename(columns={"depth": "d"})
    nd = nb[~(nb["apex_at_record_end"] | nb["apex_on_trace_seam"])]
    stage_items["not_D"] = nd[["activity_id", "path", "depth"]].rename(columns={"depth": "d"})
    ne = nd[nd["qc_verdict"] != "unknown"]
    assert set(ne["row"]) == set(ok[~ok["qc_excluded"]]["row"]), "B->D->E chain must equal qc_excluded"
    stage_items["not_E"] = ne[["activity_id", "path", "depth"]].rename(columns={"depth": "d"})
    sup = est_df[est_df["kept_by_support"]]
    stage_items["supported"] = pd.DataFrame(
        {"activity_id": sup["activity_id"], "path": "cluster", "d": sup["depth_m"]}
    )
    # nominal-vs-fitted depth offset of generator candidates (for the stage change)
    off = (ok["depth"] - ok["depth_nom"]).dropna()
    repro["fit_minus_nominal_depth_m_q"] = [round(float(x), 4) for x in np.quantile(off, [0.05, 0.5, 0.95])]

    by_act = {s: {a: df for a, df in items.groupby("activity_id")} for s, items in stage_items.items()}

    def evidence(stage: str, act: str, d: float) -> tuple[int, int]:
        df = by_act[stage].get(act)
        if df is None:
            return 0, 0
        near = df[np.abs(df["d"] - d) <= LIMIT]
        return len(near), near["path"].nunique()

    def coverage(stage: str, act: str) -> float:
        df = by_act[stage].get(act)
        if df is None or len(df) == 0:
            return 0.0
        dd = df["d"].dropna().to_numpy()
        return float(np.mean(np.any(np.abs(GRID[:, None] - dd[None, :]) <= LIMIT, axis=1)))

    for s in STAGES:
        ev_counts = [evidence(s, a, d) for a, d in zip(truth["activity_id"], truth["gt_depth_m"])]
        truth[f"n_{s}"] = [c[0] for c in ev_counts]
        truth[f"lines_{s}"] = [c[1] for c in ev_counts]
        truth[f"cov_{s}"] = [coverage(s, a) for a in truth["activity_id"]]

    def loss_stage(row, min_lines: int) -> str:
        for s in STAGES:
            need = 1 if s == "supported" else min(min_lines, max(1, row["n_lines_act"]))
            if row[f"lines_{s}"] < need:
                return s
        return "final"

    truth["loss_stage"] = [
        "pass" if r["status"] == "pass" else loss_stage(r, 1) for _, r in truth.iterrows()
    ]
    truth["loss_stage_2lines"] = [
        "pass" if r["status"] == "pass" else loss_stage(r, 2) for _, r in truth.iterrows()
    ]

    # generator reasons near the truth for gen1 losses
    reason_near = []
    for _, r in truth.iterrows():
        df = gi_[(gi_["activity_id"] == r["activity_id"]) & (np.abs(gi_["depth_nom_m"] - r["gt_depth_m"]) <= LIMIT)]
        reason_near.append(";".join(f"{k}={v}" for k, v in sorted(Counter(df["reason"].fillna("").replace("", "ok")).items())))
    truth["gen_reasons_near"] = reason_near

    # record window per activity: max over lines of the observable depth before the D record end
    gl = gen_lines.copy()
    for c in ("n_samples", "dt_ns", "t_zero_ns", "v_meta", "depth_window_m", "depth_mute_m"):
        gl[c] = pd.to_numeric(gl[c], errors="coerce")
    gl["depth_record_end_m"] = gl["v_meta"] * ((gl["n_samples"] - 1) * gl["dt_ns"] - RECORD_END_NS - gl["t_zero_ns"]) / 2
    win = gl.groupby("activity_id").agg(
        depth_record_end_max=("depth_record_end_m", "max"),
        depth_mute_min=("depth_mute_m", "min"),
        cap_lines=("cap_active", lambda s: int(as_bool(s.astype(str)).sum())),
        n_gen_lines=("path", "count"),
        eps_meta=("v_meta", lambda s: round(float((0.299792458 / s.median()) ** 2), 2)),
    )
    truth = truth.merge(win, left_on="activity_id", right_index=True, how="left")
    truth["beyond_record_end"] = truth["gt_depth_m"] > truth["depth_record_end_max"]
    truth["eps_header_default"] = np.isclose(truth["eps_meta"], 9.0)

    # (iv) truth-side flags
    deg = []
    for _, r in truth.iterrows():
        others = truth[(truth["activity_id"] == r["activity_id"]) & (truth["gt_index"] != r["gt_index"])]
        deg.append(int((np.abs(others["gt_depth_m"] - r["gt_depth_m"]) <= ev.CLUSTER_WIDTH_M + 1e-9).sum()))
    truth["n_gt_same_depth_010"] = deg
    gtr = gt_rows.copy()
    gtr["gt_index"] = gtr.groupby("activity_id").cumcount()
    truth = truth.merge(
        gtr[["activity_id", "gt_index", "distance_m", "note", "trench_no"]],
        on=["activity_id", "gt_index"],
        how="left",
    )
    truth["distance_is_range"] = truth["distance_m"].astype(str).str.contains("-", regex=False)
    truth["center_rescues"] = [
        (r["status"] == "failed") and abs(r["error_m"] + r["gt_center_shift_m"]) <= LIMIT
        for _, r in truth.iterrows()
    ]

    # mechanism class for non-passing truths
    def mech(r) -> str:
        if r["status"] == "pass":
            return "pass"
        if r["beyond_record_end"]:
            return "iv-window truth below record end"
        return LOSS_LABEL[r["loss_stage"]]

    truth["mechanism"] = [mech(r) for _, r in truth.iterrows()]

    # sub-classes
    owner = {}
    for _, r in truth.iterrows():
        if pd.notna(r["est_index"]):
            owner[(r["activity_id"], int(r["est_index"]))] = r["status"]

    def multiplicity(row) -> tuple[int, bool, float]:
        per = defaultdict(list)
        for p, t in zip(row["member_paths"].split(), row["member_traces"].split()):
            per[p].append(int(t))
        counts = [len(v) for v in per.values()]
        lateral = any(len(v) > 1 and (max(v) - min(v)) * 0.02 > gen_nms_m for v in per.values())
        return max(counts), lateral, float(np.median(counts))

    gen_nms_m = 0.3  # generator NMS distance (fourtu_apex_candidate_select.NMS_M)
    mult = [multiplicity(r) for _, r in est_df.iterrows()]
    est_df["max_per_line"] = [m[0] for m in mult]
    est_df["lateral_multi"] = [m[1] for m in mult]
    est_df["median_per_line"] = [m[2] for m in mult]
    nd_unknown = nd[nd["qc_verdict"] == "unknown"]
    subs = []
    for _, r in truth.iterrows():
        m = r["mechanism"]
        act = r["activity_id"]
        if m == LOSS_LABEL["final"]:
            e = estimates[act]
            near = [i for i, x in enumerate(e) if abs(x - r["gt_depth_m"]) <= LIMIT]
            owners = [owner.get((act, i)) for i in near]
            if all(o == "pass" for o in owners):
                lat = bool(
                    est_df[(est_df["activity_id"] == act) & est_df["est_index"].isin(near)]["lateral_multi"].any()
                )
                s = "ii-a shared estimate" + (" (lateral multi)" if lat else "")
            else:
                s = "ii-b assignment choice"
        elif m == LOSS_LABEL["supported"]:
            s = "i-sup-a one line only" if r["lines_not_E"] <= 1 else "i-sup-b >=2 lines, no cluster within 0.30 m"
        elif m == LOSS_LABEL["not_E"]:
            df = nd_unknown[(nd_unknown["activity_id"] == act) & (np.abs(nd_unknown["depth"] - r["gt_depth_m"]) <= LIMIT)]
            s = "i-qcE " + (Counter(df["qc_reason"]).most_common(1)[0][0] if len(df) else "?")
        else:
            s = m
        subs.append(s)
    truth["sub_mechanism"] = subs
    dbin = pd.cut(truth["gt_depth_m"], [0.0, 0.5, 0.8, 1.1, 1.4, 6.0])
    depth_by_mech = pd.crosstab(dbin.astype(str), truth["mechanism"]).to_dict(orient="index")
    depth_by_status = pd.crosstab(dbin.astype(str), truth["status"]).to_dict(orient="index")
    sup_cl = est_df[est_df["kept_by_support"]]
    labelfree_mult = {
        "supported_clusters": int(len(sup_cl)),
        "with_lateral_multi": int(sup_cl["lateral_multi"].sum()),
        "extra_estimates_if_median_per_line": int(np.floor(sup_cl["median_per_line"]).sum() - len(sup_cl)),
        "extra_estimates_if_max_per_line": int(sup_cl["max_per_line"].sum() - len(sup_cl)),
    }

    # ------------------------------------------------------------ step 3: failed pairs (iii)
    rev = review.copy()
    rev["trace"] = rev["trace"].astype(int)
    rev_key = {}
    for _, r in rev.iterrows():
        rev_key.setdefault((r["activity_id"], r["path"]), []).append((r["trace"], float(r["sample"]), r["type"], r["blind_id"]))
    fp_rows = []
    for _, r in truth[truth["status"] == "failed"].iterrows():
        act = r["activity_id"]
        e = estimates[act]
        est = e[int(r["est_index"])]
        cl = est_df[(est_df["activity_id"] == act) & (est_df["est_index"] == r["est_index"])].iloc[0]
        others = truth[(truth["activity_id"] == act) & (truth["gt_index"] != r["gt_index"])]
        near_other = others[np.abs(others["gt_depth_m"] - est) <= LIMIT]
        other_est = [x for i, x in enumerate(e) if i != int(r["est_index"])]
        nearest_other_est = min((abs(x - est) for x in other_est), default=np.nan)
        # reviewed member labels
        labels = []
        for row_id, path, tr in zip(cl["member_rows"].split(), cl["member_paths"].split(), cl["member_traces"].split()):
            c = cand[cand["row"] == int(row_id)].iloc[0]
            for t, s, typ, bid in rev_key.get((act, path), []):
                if abs(t - int(tr)) <= 3 and abs(s - c["candidate_sample"]) <= 5:
                    labels.append(f"{bid}:{typ}")
        if len(near_other):
            ok_other = near_other[near_other["status"] == "pass"]
            sub = "iii-a second event of a listed pipe (paired pipe passes)" if len(ok_other) else "iii-a' near another listed pipe"
        elif est < NEAR_SURFACE_M:
            sub = "iii-c near-surface event (< 0.25 m, unlisted)"
        else:
            sub = "iii-b unlisted event"
        fp_rows.append(
            {
                "activity_id": act,
                "site": r["site"],
                "gt_index": r["gt_index"],
                "gt_depth_m": r["gt_depth_m"],
                "code": r["code"],
                "est_depth_m": round(est, 4),
                "error_m": round(r["error_m"], 4),
                "ratio": round(est / r["gt_depth_m"], 3),
                "side": "shallow" if r["error_m"] < 0 else "deep",
                "n_gt_act": r["n_gt_act"],
                "n_est_act": r["n_est_act"],
                "est_n_lines": cl["n_lines"],
                "est_n_members": cl["n_members"],
                "est_member_qc": cl["member_qc"],
                "est_member_sources": cl["member_sources"],
                "est_near_other_gt": ";".join(f"{x:.2f}({s})" for x, s in zip(near_other["gt_depth_m"], near_other["status"])),
                "nearest_other_est_m": round(nearest_other_est, 4) if np.isfinite(nearest_other_est) else None,
                "review_labels": " ".join(labels),
                "paired_event_class": sub,
                "truth_loss_stage": r["loss_stage"],
                "truth_loss_stage_2lines": r["loss_stage_2lines"],
                "truth_mechanism": r["mechanism"],
                "truth_sub_mechanism": r["sub_mechanism"],
                "est_max_per_line": cl["max_per_line"],
                "est_lateral_multi": cl["lateral_multi"],
                "center_rescues": r["center_rescues"],
                "gt_center_shift_m": round(r["gt_center_shift_m"], 4),
                "n_gt_same_depth_010": r["n_gt_same_depth_010"],
                "beyond_record_end": r["beyond_record_end"],
                "eps_meta": r["eps_meta"],
            }
        )
    fp = pd.DataFrame(fp_rows)

    # ------------------------------------------------------------ 18 activities with no estimate
    za = []
    for act, grp in truth.groupby("activity_id"):
        if grp["n_est_act"].iloc[0] != 0:
            continue
        a_c = cand[cand["activity_id"] == act]
        a_ok = a_c[a_c["status"] == "ok"]
        a_g = gi_[gi_["activity_id"] == act]
        cl = est_df[est_df["activity_id"] == act]
        za.append(
            {
                "activity_id": act,
                "site": ev.site_of_activity(act),
                "n_gt": len(grp),
                "gt_depths_m": " ".join(f"{d:.2f}" for d in sorted(grp["gt_depth_m"])),
                "n_lines": line_counts.get(act, 0),
                "eps_meta": grp["eps_meta"].iloc[0],
                "n_peaks": int(len(a_g)),
                "n_accepted": int(a_g["stage"].isin(["accepted", "kept"]).sum()),
                "n_kept": int(len(a_c)),
                "n_fit_ok": int(len(a_ok)),
                "n_out": int((a_ok["qc_verdict"] == "out").sum()),
                "n_unknown": int((a_ok["qc_verdict"] == "unknown").sum()),
                "n_in": int((a_ok["qc_verdict"] == "in").sum()),
                "n_D": int((a_ok["apex_at_record_end"] | a_ok["apex_on_trace_seam"]).sum()),
                "n_after_qc": int((~a_ok["qc_excluded"]).sum()),
                "n_clusters": int(len(cl)),
                "cluster_lines_max": int(cl["n_lines"].max()) if len(cl) else 0,
                "unknown_reasons": ";".join(f"{k}={v}" for k, v in sorted(Counter(a_ok[a_ok["qc_verdict"] == "unknown"]["qc_reason"]).items())),
                "bottleneck": (
                    "no generator candidate" if len(a_c) == 0
                    else "all fits failed" if len(a_ok) == 0
                    else "all removed by QC" if int((~a_ok["qc_excluded"]).sum()) == 0
                    else "single-line clusters only (support)"
                ),
                "truth_loss_stages": ";".join(f"{k}={v}" for k, v in sorted(Counter(grp["loss_stage"]).items())),
            }
        )
    zero_acts = pd.DataFrame(za)

    # ------------------------------------------------------------ step 4: label-free depth dependence
    bins = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 3.0]
    lab = pd.cut(gi_["depth_nom_m"], bins)
    lf = []
    for b, grp in gi_.groupby(lab, observed=False):
        n = len(grp)
        acc = grp["stage"].isin(["accepted", "kept"]).sum()
        kp = (grp["stage"] == "kept").sum()
        lf.append({"stage": "generator", "depth_bin_m": str(b), "n_peak": n, "n_accepted": int(acc), "n_kept": int(kp),
                   "accept_rate": round(acc / n, 4) if n else None, "kept_of_accepted": round(kp / acc, 4) if acc else None,
                   "nms_or_cap_cut": int((grp["reason"] == "nms_or_cap").sum())})
    lab2 = pd.cut(ok["depth"], bins)
    for b, grp in ok.groupby(lab2, observed=False):
        n = len(grp)
        lf.append({"stage": "qc", "depth_bin_m": str(b), "n_fit_ok": n,
                   "frac_in": round((grp["qc_verdict"] == "in").mean(), 4) if n else None,
                   "frac_out": round((grp["qc_verdict"] == "out").mean(), 4) if n else None,
                   "frac_unknown": round((grp["qc_verdict"] == "unknown").mean(), 4) if n else None,
                   "frac_D": round((grp["apex_at_record_end"] | grp["apex_on_trace_seam"]).mean(), 4) if n else None,
                   "frac_after_qc": round((~grp["qc_excluded"]).mean(), 4) if n else None,
                   "unknown_insufficient_picks": int((grp["qc_reason"] == "insufficient_picks").sum()),
                   "unknown_unstable": int((grp["qc_reason"] == "unstable").sum()),
                   "unknown_fit_failed": int((grp["qc_reason"] == "fit_failed").sum())})
    est_lab = pd.cut(est_df["depth_m"], bins)
    for b, grp in est_df.groupby(est_lab, observed=False):
        n = len(grp)
        lf.append({"stage": "support", "depth_bin_m": str(b), "n_clusters": n,
                   "frac_supported": round(grp["kept_by_support"].mean(), 4) if n else None})
    labelfree = pd.DataFrame(lf)

    # ------------------------------------------------------------ write
    truth_cols = [c for c in truth.columns if not c.startswith("cov_")]
    truth[truth_cols].to_csv(DOCS / "m210d_truth_classes.csv", index=False, float_format="%.4f")
    fp.to_csv(DOCS / "m210d_failed_pairs.csv", index=False, float_format="%.4f")
    zero_acts.to_csv(DOCS / "m210d_zero_activities.csv", index=False)
    labelfree.to_csv(DOCS / "m210d_stage_depth_labelfree.csv", index=False)
    est_df.to_csv(DATA / "m210d_estimates.csv", index=False, float_format="%.4f")
    truth.to_csv(DATA / "m210d_truth_full.csv", index=False, float_format="%.4f")

    nonpass = truth[truth["status"] != "pass"]
    summary = {
        "note": "already-seen 4TU data, exploration only (M2-10 d); not a verdict, not for tuning",
        "reproduction": repro,
        "status_counts": dict(Counter(truth["status"])),
        "mechanism_counts_all_nonpass": dict(Counter(nonpass["mechanism"])),
        "mechanism_by_status": {s: dict(Counter(g["mechanism"])) for s, g in nonpass.groupby("status")},
        "sub_mechanism_by_status": {s: dict(Counter(g["sub_mechanism"])) for s, g in nonpass.groupby("status")},
        "depth_bin_by_mechanism": depth_by_mech,
        "depth_bin_by_status": depth_by_status,
        "labelfree_cluster_multiplicity": labelfree_mult,
        "loss_stage_1line": dict(Counter(nonpass["loss_stage"])),
        "loss_stage_2lines": dict(Counter(nonpass["loss_stage_2lines"])),
        "chance_coverage_mean_nonpass": {s: round(float(nonpass[f"cov_{s}"].mean()), 3) for s in STAGES},
        "chance_coverage_mean_pass": {s: round(float(truth[truth["status"] == "pass"][f"cov_{s}"].mean()), 3) for s in STAGES},
        "evidence_rate_nonpass": {s: round(float((nonpass[f"lines_{s}"] >= 1).mean()), 3) for s in STAGES},
        "evidence_rate_nonpass_2lines": {s: round(float((nonpass[f"lines_{s}"] >= 2).mean()), 3) for s in STAGES},
        "failed_pairs": {
            "n": int(len(fp)),
            "shallow": int((fp["side"] == "shallow").sum()),
            "ratio_median_shallow": round(float(fp[fp["side"] == "shallow"]["ratio"].median()), 3) if len(fp) else None,
            "paired_event_class": dict(Counter(fp["paired_event_class"])),
            "truth_mechanism": dict(Counter(fp["truth_mechanism"])),
            "center_rescues": int(fp["center_rescues"].sum()),
            "with_review_label": int((fp["review_labels"] != "").sum()),
        },
        "truth_flags_nonpass": {
            "beyond_record_end": int(nonpass["beyond_record_end"].sum()),
            "multi_trench": int(nonpass["multi_trench"].sum()),
            "same_depth_gt_010": int((nonpass["n_gt_same_depth_010"] > 0).sum()),
            "distance_is_range": int(nonpass["distance_is_range"].sum()),
        },
        "truth_flags_pass": {
            "same_depth_gt_010": int((truth[truth["status"] == "pass"]["n_gt_same_depth_010"] > 0).sum()),
        },
        "depth_quantiles": {
            s: [round(float(x), 3) for x in np.quantile(g["gt_depth_m"], [0.1, 0.5, 0.9])]
            for s, g in truth.groupby("status")
        },
        "zero_estimate_activities": {
            "n": int(len(zero_acts)),
            "bottleneck": dict(Counter(zero_acts["bottleneck"])) if len(zero_acts) else {},
        },
    }
    (DOCS / "m210d_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
