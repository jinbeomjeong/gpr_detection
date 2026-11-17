"""Re-aggregate the finished M3-3 J1 outputs for the judgment record (read only).

Spec: docs/processing/m33_j1_preregistration.md v0.3.5 (commit 1a73ba8).
Reads the verdict / report json and the eval csv files written by
``analysis/processing/m33_j1_judge.py`` under ``docs/processing/depth_4tu_j1*``
and never writes there. Nothing is re-run: no candidate generation, no depth
fit, no baseline draw.

Checks done by hand for the judgment (verif record 7):
  * n_pass and n_pairs recounted from each matches csv (main combination),
    compared with the json values;
  * raw sha256 of the four verdict eval outputs compared with
    ``eval_outputs_sha256`` of the verdict json;
  * candidate funnel of the fold 0 post run (status, QC, clusters, support);
  * per-site and per-activity counts, the failing pairs, PP pre vs post
    candidate difference, signal-candidate comparison.

Output: analysis/processing/m33_j1_judgment_tables.json (stdout: last line
JSON with the output path only).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "processing"
OUT = REPO / "analysis" / "processing" / "m33_j1_judgment_tables.json"

PREFIXES = {
    "verdict_fold0_post": "depth_4tu_j1",
    "pre": "depth_4tu_j1_pre",
    "fold1": "depth_4tu_j1_f1",
    "fold2": "depth_4tu_j1_f2",
    "fold3": "depth_4tu_j1_f3",
    "fold4": "depth_4tu_j1_f4",
    "signal": "depth_4tu_j1_signal",
}
MAIN = {
    "gt_set": "exclude_x",
    "assumption": "top",
    "zero_variant": "detected",
    "variant": "main",
}
TOL = 0.30 + 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, value in MAIN.items():
        mask &= frame[key] == value
    return frame[mask]


def json_of(key: str, prefix: str) -> dict:
    name = f"{prefix}_verdict.json" if key.startswith("verdict") else f"{prefix}_report.json"
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", float("nan")), errors="coerce")


def recount(prefix: str) -> dict:
    matches = main_rows(read_csv(DOCS / f"{prefix}_matches.csv"))
    has_est = matches["est_depth_m"] != ""
    paired = matches[has_est & (matches["gt_depth_m"] != "")]
    within = paired["within_30cm"] == "True"
    # independent predicate from error_m (|error| <= 0.30 + 1e-12)
    err = to_float(paired["error_m"]).abs()
    return {
        "n_match_rows": int(len(matches)),
        "n_pairs_recount": int(len(paired)),
        "n_pass_recount": int(within.sum()),
        "n_pass_from_error_m": int((err <= TOL).sum()),
        "n_gt_rows": int((matches["gt_depth_m"] != "").sum()),
    }


def summary_main(prefix: str) -> dict:
    summary = read_csv(DOCS / f"{prefix}_summary.csv")
    row = main_rows(summary[summary["scope"] == "all"])
    assert len(row) == 1, prefix
    keep = [
        "n_activities", "n_gt", "n_est", "n_pairs", "n_pass", "pass_a", "pass_b",
        "mae_m", "bias_m", "missed", "false_est", "baseline_pass_b_mean",
        "baseline_pass_b_p025", "baseline_pass_b_p975", "baseline_p",
        "baseline_global_pass_b_mean", "baseline_global_p", "gate_precision",
        "gate_recall", "n_gt_unobservable", "n_gt_window_unknown",
        "n_gt_observable", "pass_b_observable",
    ]
    return {k: row.iloc[0][k] for k in keep}


def run_table() -> list[dict]:
    rows = []
    for key, prefix in PREFIXES.items():
        doc = json_of(key, prefix)
        res = doc["main_result"]
        sub = doc["substantive"]
        reach = (doc.get("report") or {}).get("reachability") or {}
        boot = (doc.get("report") or {}).get("site_bootstrap") or {}
        rows.append(
            {
                "run": key,
                "prefix": prefix,
                "kind": doc["kind"],
                "status": doc["status"],
                "candidates_csv": doc["inputs"]["candidates_csv"],
                "candidates_sha256": doc["inputs"]["candidates_sha256"],
                "boxes_sha256": doc["inputs"]["boxes_sha256"],
                "candidate_counts": doc["candidate_counts"],
                "n_pass": res["n_pass"],
                "n_gt": res["n_gt"],
                "pass_b": res["pass_b"],
                "k": res["k"],
                "p": res["p"],
                "n_pairs": res["n_pairs"],
                "n_est": res["n_est"],
                "false_est": res["false_est"],
                "missed": res["missed"],
                "pass_a": res["pass_a"],
                "baseline_pass_a_mean": res["baseline_pass_a_mean"],
                "mae_m": res["mae_m"],
                "bias_m": res["bias_m"],
                "b_max": res["b_max"],
                "baseline_pass_b_mean": sub["baseline_pass_b_mean"],
                "baseline_pass_b_p025": sub["baseline_pass_b_p025"],
                "baseline_pass_b_p975": sub["baseline_pass_b_p975"],
                "delta_b": sub["delta_b"],
                "verdict_case": (doc.get("verdict") or {}).get("case"),
                "k_perfect": reach.get("k_perfect"),
                "t_50": reach.get("t_50"),
                "n_req": reach.get("n_req"),
                "n_req_exceeds_n_pairs": reach.get("n_req_exceeds_n_pairs"),
                "excess_estimates": reach.get("excess_estimates"),
                "n_act_more_est": reach.get("n_activities_more_estimates_than_gt"),
                "n_single_gt_with_est": reach.get("n_single_gt_activities_with_estimate"),
                "bootstrap": boot,
                "report_failure": doc.get("report_failure"),
                "recount": recount(prefix),
                "summary_main": summary_main(prefix),
                "depth_run_diagnostics": doc.get("depth_run_diagnostics"),
            }
        )
    return rows


def eval_sha_check() -> dict:
    doc = json_of("verdict", "depth_4tu_j1")
    out = {}
    for name, expected in doc["eval_outputs_sha256"]["raw"].items():
        measured = sha256(DOCS / name)
        out[name] = {"expected": expected, "measured": measured, "same": measured == expected}
    out["verdict_json_sha256"] = sha256(DOCS / "depth_4tu_j1_verdict.json")
    out["verdict_md_sha256"] = sha256(DOCS / "depth_4tu_j1_verdict.md")
    out["candidates_input_sha256"] = sha256(DOCS / "depth_4tu_j1_candidates_input.csv")
    return out


def funnel(prefix: str) -> dict:
    cand = read_csv(DOCS / f"{prefix}_candidates.csv")
    clusters = read_csv(DOCS / f"{prefix}_clusters.csv")
    det = clusters[clusters["zero_variant"] == "detected"]
    qc_in = cand[(cand["status"] == "ok") & (cand["qc_excluded"] == "False")]
    lines_per_act = qc_in.groupby("activity_id")["path"].nunique()
    det_n_lines = pd.to_numeric(det["n_lines"])
    return {
        "n_candidates": int(len(cand)),
        "status": cand["status"].value_counts().to_dict(),
        "qc_verdict": cand["qc_verdict"].value_counts().to_dict(),
        "qc_reason_top": cand["qc_reason"].replace("", "(blank)").value_counts().head(12).to_dict(),
        "qc_excluded": cand["qc_excluded"].value_counts().to_dict(),
        "n_qc_kept": int(len(qc_in)),
        "n_activities_with_candidates": int(cand["activity_id"].nunique()),
        "n_activities_with_qc_kept": int(qc_in["activity_id"].nunique()),
        "n_activities_qc_kept_one_line_only": int((lines_per_act == 1).sum()),
        "clusters_zero_variants": clusters["zero_variant"].value_counts().to_dict(),
        "detected_clusters": int(len(det)),
        "detected_kept_by_support": int((det["kept_by_support"] == "True").sum()),
        "detected_cluster_n_lines": det_n_lines.value_counts().sort_index().to_dict(),
        "detected_activities_with_kept": int(
            det[det["kept_by_support"] == "True"]["activity_id"].nunique()
        ),
        "apex_on_trace_seam": cand["apex_on_trace_seam"].value_counts().to_dict(),
        "candidate_moved": cand["candidate_moved"].value_counts().to_dict(),
        "label_est": cand["label_est"].value_counts().to_dict(),
        "zero_rule": cand["zero_rule"].value_counts().to_dict(),
    }


def per_site_and_failures(prefix: str) -> dict:
    matches = main_rows(read_csv(DOCS / f"{prefix}_matches.csv"))
    matches = matches[matches["gt_depth_m"] != ""].copy()
    matches["has_est"] = matches["est_depth_m"] != ""
    matches["pass"] = matches["within_30cm"] == "True"
    # matches csv holds the matched pairs only (missed ground truths are not
    # rows), so these are pair counts, not N_g,s (see summary_by_site).
    site = (
        matches.groupby("site")
        .agg(n_pair_rows=("gt_depth_m", "size"), n_pass=("pass", "sum"))
        .reset_index()
    )
    site["site_int"] = site["site"].astype(int)
    site = site.sort_values("site_int").drop(columns="site_int")
    act = (
        matches.groupby(["site", "activity_id"])
        .agg(n_pair_rows=("gt_depth_m", "size"), n_pass=("pass", "sum"))
        .reset_index()
    )
    fails = matches[matches["has_est"] & ~matches["pass"]][
        ["activity_id", "site", "est_depth_m", "gt_depth_m", "gt_code", "error_m"]
    ]
    passes = matches[matches["pass"]]
    err_pass = to_float(passes["error_m"])
    gt_all = to_float(matches["gt_depth_m"])
    gt_paired = to_float(matches.loc[matches["has_est"], "gt_depth_m"])
    return {
        "by_site": site.to_dict(orient="records"),
        "n_activities_with_pair": int(len(act)),
        "n_activities_with_pass": int((act["n_pass"] > 0).sum()),
        "activities_with_pair": act.to_dict(orient="records"),
        "failing_pairs": fails.to_dict(orient="records"),
        "pass_error_mean_m": float(err_pass.mean()) if len(err_pass) else None,
        "gt_depth_median_all_m": float(gt_all.median()),
        "gt_depth_median_paired_m": float(gt_paired.median()) if len(gt_paired) else None,
        "gt_code_all": matches["gt_code"].value_counts().to_dict(),
        "gt_code_paired": matches.loc[matches["has_est"], "gt_code"].value_counts().to_dict(),
    }


def summary_by_site(prefix: str) -> list[dict]:
    """Main-combination summary rows of each site scope (N_g,s, n_est, ...)."""
    summary = read_csv(DOCS / f"{prefix}_summary.csv")
    rows = main_rows(summary[summary["scope"].str.startswith("site_")])
    keep = ["scope", "n_activities", "n_gt", "n_est", "n_pairs", "n_pass", "pass_b",
            "mae_m", "bias_m", "missed", "false_est", "baseline_pass_b_mean", "baseline_p"]
    out = rows[keep].copy()
    out["site_int"] = out["scope"].str.replace("site_", "", regex=False).astype(int)
    out = out.sort_values("site_int").drop(columns="site_int")
    return out.to_dict(orient="records")


def combos(prefix: str) -> list[dict]:
    """[U3] S1-like table: top/centre x three zero variants, exclude_x, main."""
    summary = read_csv(DOCS / f"{prefix}_summary.csv")
    rows = summary[
        (summary["scope"] == "all")
        & (summary["gt_set"] == "exclude_x")
        & (summary["variant"] == "main")
    ]
    keep = ["assumption", "zero_variant", "n_est", "n_pairs", "n_pass", "pass_b",
            "mae_m", "bias_m", "baseline_pass_b_mean", "baseline_p"]
    return rows[keep].to_dict(orient="records")


def kept_clusters_by_site(prefix: str) -> dict:
    clusters = read_csv(DOCS / f"{prefix}_clusters.csv")
    det = clusters[(clusters["zero_variant"] == "detected") & (clusters["kept_by_support"] == "True")]
    return det.groupby("site").size().to_dict()


def pre_vs_post() -> dict:
    cols = ["activity_id", "path", "trace", "sample", "tau_ns"]
    post = read_csv(DOCS / "depth_4tu_j1_candidates_input.csv")
    pre = read_csv(DOCS / "depth_4tu_j1_pre_candidates_input.csv")
    same_core = post[cols].reset_index(drop=True).equals(pre[cols].reset_index(drop=True))
    other = [c for c in post.columns if c not in cols]
    differing = {c: int((post[c].values != pre[c].values).sum()) for c in other} if len(post) == len(pre) else None
    actions = post["note"].str.extract(r"pp_action=([a-z_]+)")[0].value_counts().to_dict()
    return {
        "n_post": int(len(post)),
        "n_pre": int(len(pre)),
        "core_columns_identical": bool(same_core),
        "differing_other_columns": differing,
        "post_pp_action_counts": actions,
    }


def main() -> int:
    started = time.time()
    result = {
        "spec": "docs/processing/m33_j1_preregistration.md v0.3.5 (1a73ba8)",
        "runs": run_table(),
        "eval_sha_check": eval_sha_check(),
        "funnel_fold0_post": funnel("depth_4tu_j1"),
        "funnel_signal": funnel("depth_4tu_j1_signal"),
        "sites_fold0_post": per_site_and_failures("depth_4tu_j1"),
        "sites_signal": per_site_and_failures("depth_4tu_j1_signal"),
        "summary_by_site": {
            key: summary_by_site(prefix) for key, prefix in PREFIXES.items()
        },
        "combos": {key: combos(prefix) for key, prefix in PREFIXES.items()},
        "kept_clusters_by_site": {
            "verdict_fold0_post": kept_clusters_by_site("depth_4tu_j1"),
            "signal": kept_clusters_by_site("depth_4tu_j1_signal"),
        },
        "pre_vs_post": pre_vs_post(),
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(REPO)), "elapsed_s": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
