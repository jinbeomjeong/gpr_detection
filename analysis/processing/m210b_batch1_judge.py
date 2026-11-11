"""M2-10 (b) batch-1 judgment of the selector-floor rules (gprMax S4N/S6P, sealed criteria).

Criteria: docs/processing/m210b_selector_floor.md v1.4 (seal commit bc9a474) section 4-2,
decision ranks 1-9; pre-result decisions D1, D2, P-F1-P-F3 in
docs/processing/m210b_harness_interpretation.md (commit 7e843d6). This script only
counts; it does not change any criterion. No 4TU truth is read.

Inputs (read only): data/processing/m210b_s4s6/runs.csv, run_info.json, pre_run.txt,
post_run.txt.
Outputs: docs/processing/m210b_batch1_judgment.json (all counts and the rank).

Run: conda run -n gpr-dev python -I analysis/processing/m210b_batch1_judge.py
"""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data" / "processing" / "m210b_s4s6"
OUT = ROOT / "docs" / "processing" / "m210b_batch1_judgment.json"
SEAL = "bc9a474"
INTERP_COMMIT = "7e843d6"
SEALED_DOC = "docs/processing/m210b_selector_floor.md"
INTERP_DOC = "docs/processing/m210b_harness_interpretation.md"

CANDIDATES = ("B_full", "B_min", "A_skip1", "C_none", "B_noise")
RULES = ("current",) + CANDIDATES
DELTAS = (-1.0, 0.0, 1.0, 2.35)
PAIRS = ((0.0, 2.35), (-1.0, 2.35))
RHOS = (10, 20)
N_SEEDS = 10
TOL = 1e-9  # rates are k/10; exact comparisons with float tolerance
RATE_GAP_MAX = 0.1  # C, F1, F2, F3 (sealed)
EFFECT_MIN = 0.5  # R (i) (sealed)
LAMBDA_MIN = 10.0  # G (sealed)
RHO_MIN = 5.0  # Q (sealed)
FLOOR_SEEDS_MIN = 5  # G (sealed)
F4_MARGIN = 0.5  # F4, F4b (sealed)
F5_TRACES, F5_SAMPLES = 1, 0.5  # F5 (sealed)

PIPE_MODELS = (
    "S4N_D12_S010", "S6P_D12_S010", "S4N_D15_S010", "S6P_D15_S010",
    "S4N_D15_S030", "S6P_D15_S030", "S4N_D15_S040", "S6P_D15_S040",
)
NP_MODEL = "S6P_NP_S010"
S4N_OF = {m: m.replace("S6P_", "S4N_") for m in PIPE_MODELS if m.startswith("S6P_")}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def as_bool(s: pd.Series) -> pd.Series:
    """'True' -> True; anything else (False, blank) -> False (P-F1: blank = not satisfied)."""
    return s.astype(str).str.strip() == "True"


def as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", float("nan")), errors="coerce")


def load():
    runs = pd.read_csv(RUN_DIR / "runs.csv", dtype=str, keep_default_na=False)
    info = json.loads((RUN_DIR / "run_info.json").read_text(encoding="utf-8"))
    runs["delta"] = as_float(runs["delta_ns"]).round(4)
    runs["rho_i"] = as_float(runs["rho"]).astype(int)
    runs["dt"] = as_float(runs["dt_ns"])
    runs["ramp"] = runs["ramp_noise"].astype(str).str.strip() == "1"
    for c in ("accepted_apex", "kept_apex", "floor_held_at_apex"):
        runs[c + "_b"] = as_bool(runs[c])
    for c in ("rho_real", "rho_real_cal", "Lambda", "n_false_deep", "kept_apex_trace",
              "kept_apex_sample", "argmax_level_periods"):
        runs[c + "_f"] = as_float(runs[c])
    return runs, info


def pf1_checks(runs: pd.DataFrame, info: dict) -> dict:
    out = {}
    out["n_runs_total"] = int(len(runs))
    out["n_runs_info"] = int(info.get("n_runs", -1))
    out["n_runs_without_zero"] = int(info.get("n_runs_without_zero", -1))
    J = runs[(runs["dt"].sub(0.098).abs() < 1e-12) & ~runs["ramp"]]
    cells = J.groupby(["model_id", "rho_i", "rule", "delta"])["seed"].agg(["count", "nunique"])
    expected = {(m, r, u, d) for m in PIPE_MODELS + (NP_MODEL,) for r in RHOS for u in RULES for d in DELTAS}
    got = set(cells.index)
    out["judgment_rows"] = int(len(J))
    out["cells_expected"] = len(expected)
    out["cells_missing"] = sorted(map(str, expected - got))
    out["cells_extra"] = sorted(map(str, got - expected))
    out["cells_not_10_seeds"] = [str(k) for k, v in cells.iterrows() if not (v["count"] == N_SEEDS and v["nunique"] == N_SEEDS)]
    R = runs[runs["ramp"] & (runs["dt"].sub(0.098).abs() < 1e-12)]
    rc = R.groupby(["model_id", "rho_i", "rule", "delta"])["seed"].agg(["count", "nunique"])
    out["ramp_cells"] = int(len(rc))
    out["ramp_cells_not_10"] = [str(k) for k, v in rc.iterrows() if not (v["count"] == N_SEEDS and v["nunique"] == N_SEEDS)]
    out["ramp_models"] = sorted(set(R["model_id"]))
    bad_models = []
    for m in PIPE_MODELS:
        h = info["models"][m]["harness"]
        if len(h.get("L_noise_free_current") or {}) != 4 or h.get("apex_trace") is None:
            bad_models.append(m)
    out["pipe_models_bad_L_or_apex"] = bad_models
    seal = info["seal"]
    out["sealed_files"] = len(seal["sealed_files"])
    out["sealed_files_all_match"] = all(v["matches"] for v in seal["sealed_files"].values())
    out["dirty_files"] = len(seal["dirty_files"])
    ok = (
        out["n_runs_total"] == 4740 and out["n_runs_info"] == 4740 and out["n_runs_without_zero"] == 0
        and not out["cells_missing"] and not out["cells_extra"] and not out["cells_not_10_seeds"]
        and out["ramp_cells"] == 24 and not out["ramp_cells_not_10"] and out["ramp_models"] == [NP_MODEL]
        and not bad_models and out["sealed_files_all_match"] and out["dirty_files"] == 0
    )
    out["pass"] = bool(ok)
    return out, J, R


def rate(J, model, rho, rule, delta) -> float:
    g = J[(J["model_id"] == model) & (J["rho_i"] == rho) & (J["rule"] == rule) & (J["delta"] == delta)]
    return float(g["accepted_apex_b"].sum()) / N_SEEDS


def cell(J, model, rho, rule, delta):
    return J[(J["model_id"] == model) & (J["rho_i"] == rho) & (J["rule"] == rule) & (J["delta"] == delta)]


def r_table(J, info, use_cal: bool) -> list[dict]:
    """Q, G, C, e per S6P condition and pair (current rule)."""
    rows = []
    for m in S4N_OF:
        h = info["models"][m]["harness"]
        for rho in RHOS:
            for de, dl in PAIRS:
                ce, cl = cell(J, m, rho, "current", de), cell(J, m, rho, "current", dl)
                rho_col = "rho_real_cal_f" if use_cal else "rho_real_f"
                q = bool((ce[rho_col] >= RHO_MIN - TOL).all() and (cl[rho_col] >= RHO_MIN - TOL).all()
                         and len(ce) == N_SEEDS and len(cl) == N_SEEDS)
                if use_cal:
                    key = f"{de:+.2f}"
                    lam = h["L_noise_free_current"][key] / h["calibration"]["a0"]
                else:
                    lam_vals = ce["Lambda_f"].dropna().unique()
                    lam = float(lam_vals[0]) if len(lam_vals) == 1 and ce["Lambda_f"].notna().all() else float("nan")
                n_floor = int(ce["floor_held_at_apex_b"].sum())
                g = bool(lam > LAMBDA_MIN and n_floor >= FLOOR_SEEDS_MIN) if not math.isnan(lam) else False
                s4n = S4N_OF[m]
                c_gap = rate(J, s4n, rho, "current", dl) - rate(J, s4n, rho, "current", de)
                c = abs(c_gap) <= RATE_GAP_MAX + TOL
                e = rate(J, m, rho, "current", dl) - rate(J, m, rho, "current", de)
                rows.append({
                    "model": m, "rho": rho, "delta_e": de, "delta_l": dl,
                    "rate_e": rate(J, m, rho, "current", de), "rate_l": rate(J, m, rho, "current", dl),
                    "e": round(e, 6), "Q": q, "rho_min": float(min(ce[rho_col].min(), cl[rho_col].min())),
                    "Lambda": round(lam, 4) if not math.isnan(lam) else None, "n_floor_held_e": n_floor,
                    "G": g, "S4N_gap": round(c_gap, 6), "C": c,
                    "R_pair": bool(q and g and c and e >= EFFECT_MIN - TOL),
                })
    return rows


def f_checks(J, R, rt, x: str) -> dict:
    res = {}
    # F1 all pipe conditions
    f1 = []
    for m in PIPE_MODELS:
        for rho in RHOS:
            r0 = rate(J, m, rho, x, 0.0)
            for d in (-1.0, 1.0, 2.35):
                gap = rate(J, m, rho, x, d) - r0
                if abs(gap) > RATE_GAP_MAX + TOL:
                    f1.append({"model": m, "rho": rho, "delta": d, "rate_0": r0, "gap": round(gap, 6)})
    res["F1_fail"] = f1
    # F2 on R pairs
    f2 = []
    for row in rt:
        if row["R_pair"]:
            rx = rate(J, row["model"], row["rho"], x, row["delta_e"])
            ref = rate(J, row["model"], row["rho"], "current", row["delta_l"])
            if rx < ref - RATE_GAP_MAX - TOL:
                f2.append({**{k: row[k] for k in ("model", "rho", "delta_e", "delta_l")}, "rate_x_e": rx, "rate_cur_l": ref})
    res["F2_fail"] = f2
    # F3 S4N at delta 0
    f3 = []
    for m in PIPE_MODELS:
        if m.startswith("S4N_"):
            for rho in RHOS:
                rx, rc = rate(J, m, rho, x, 0.0), rate(J, m, rho, "current", 0.0)
                if rx < rc - RATE_GAP_MAX - TOL:
                    f3.append({"model": m, "rho": rho, "rate_x": rx, "rate_cur": rc})
    res["F3_fail"] = f3
    # F4 NP + S4N at delta 0, per condition
    f4 = []
    for m in (NP_MODEL,) + tuple(p for p in PIPE_MODELS if p.startswith("S4N_")):
        for rho in RHOS:
            mx = cell(J, m, rho, x, 0.0)["n_false_deep_f"].fillna(0).mean()
            mc = cell(J, m, rho, "current", 0.0)["n_false_deep_f"].fillna(0).mean()
            if mx > mc + F4_MARGIN + TOL:
                f4.append({"model": m, "rho": rho, "mean_x": mx, "mean_cur": mc})
    res["F4_fail"] = f4
    # F4b ramp, judged at delta 0 (as F4); other deltas reported
    f4b = {}
    for d in DELTAS:
        mx = R[(R["rule"] == x) & (R["delta"] == d)]["n_false_deep_f"].fillna(0).mean()
        mc = R[(R["rule"] == "current") & (R["delta"] == d)]["n_false_deep_f"].fillna(0).mean()
        f4b[f"{d:+.2f}"] = {"mean_x": round(float(mx), 3), "mean_cur": round(float(mc), 3),
                            "pass": bool(mx <= mc + F4_MARGIN + TOL)}
    res["F4b"] = f4b
    res["F4b_fail"] = [] if f4b["+0.00"]["pass"] else [f4b["+0.00"]]
    res["F4b_delta_dependent"] = len({v["pass"] for v in f4b.values()}) > 1
    # F5 kept-apex position, all pipe models, all deltas
    a = J[J["rule"] == x][["model_id", "rho_i", "delta", "seed", "kept_apex_b", "kept_apex_trace_f", "kept_apex_sample_f"]]
    b = J[J["rule"] == "current"][["model_id", "rho_i", "delta", "seed", "kept_apex_b", "kept_apex_trace_f", "kept_apex_sample_f"]]
    mrg = a.merge(b, on=["model_id", "rho_i", "delta", "seed"], suffixes=("_x", "_c"))
    both = mrg[mrg["kept_apex_b_x"] & mrg["kept_apex_b_c"]]
    dtr = (both["kept_apex_trace_f_x"] - both["kept_apex_trace_f_c"]).abs()
    dsm = (both["kept_apex_sample_f_x"] - both["kept_apex_sample_f_c"]).abs()
    bad = both[(dtr > F5_TRACES + TOL) | (dsm > F5_SAMPLES + TOL)]
    res["F5_pairs"] = int(len(both))
    res["F5_max_dtrace"] = float(dtr.max()) if len(both) else None
    res["F5_max_dsample"] = float(dsm.max()) if len(both) else None
    res["F5_fail"] = int(len(bad))
    res["pass_all"] = not (f1 or f2 or f3 or f4 or res["F4b_fail"] or res["F5_fail"])
    return res


def f5_diagnostic(J, info, x: str) -> dict:
    """DIAGNOSTIC ONLY (not a criterion): F5-failing pairs versus the true apex.

    For pairs where both ``x`` and current have a kept apex and F5 fails, compare the
    distance of each rule's reported kept apex to the true apex (apex trace, t_a0).
    """
    cols = ["model_id", "rho_i", "delta", "seed", "kept_apex_b", "kept_apex_trace_f", "kept_apex_sample_f"]
    a = J[J["rule"] == x][cols]
    b = J[J["rule"] == "current"][cols]
    m = a.merge(b, on=["model_id", "rho_i", "delta", "seed"], suffixes=("_x", "_c"))
    m = m[m["kept_apex_b_x"] & m["kept_apex_b_c"]].copy()
    dt = {k: info["models"][k]["harness"]["dt_ns"] for k in PIPE_MODELS}
    m["dt"] = m["model_id"].map(dt)
    m["apex_tr"] = m["model_id"].map({k: info["models"][k]["harness"]["apex_trace"] for k in PIPE_MODELS})
    m["t_a0"] = m["model_id"].map({k: info["models"][k]["harness"]["t_a0_ns"] for k in PIPE_MODELS})
    fail = m[((m["kept_apex_trace_f_x"] - m["kept_apex_trace_f_c"]).abs() > F5_TRACES + TOL)
             | ((m["kept_apex_sample_f_x"] - m["kept_apex_sample_f_c"]).abs() > F5_SAMPLES + TOL)].copy()
    for s in ("x", "c"):
        fail[f"et_{s}"] = (fail[f"kept_apex_sample_f_{s}"] * fail["dt"] - fail["t_a0"]).abs()
        fail[f"etr_{s}"] = (fail[f"kept_apex_trace_f_{s}"] - fail["apex_tr"]).abs()
    return {
        "note": "diagnostic only, not part of the sealed criteria",
        "n_fail_pairs": int(len(fail)),
        "n_realisations": int(fail.groupby(["model_id", "rho_i", "seed"]).ngroups),
        "models": sorted(set(fail["model_id"])),
        "median_abs_t_err_ns_x_cur": [round(float(fail["et_x"].median()), 4), round(float(fail["et_c"].median()), 4)],
        "x_closer_in_time": int((fail["et_x"] < fail["et_c"] - TOL).sum()),
        "cur_closer_in_time": int((fail["et_c"] < fail["et_x"] - TOL).sum()),
        "median_abs_trace_err_x_cur": [float(fail["etr_x"].median()), float(fail["etr_c"].median())],
    }


def f5_by_scope(J, rt, x: str) -> dict:
    """REPORT ONLY: F5 failures of ``x`` counted over different scopes (same-Delta pairs).

    The judgment scope is all pipe models x 4 Delta x 2 rho (``f_checks``); the other
    scopes are disclosure items requested by the post-judgment audit.
    """
    cols = ["model_id", "rho_i", "delta", "seed", "kept_apex_b", "kept_apex_trace_f", "kept_apex_sample_f"]
    a = J[J["rule"] == x][cols]
    b = J[J["rule"] == "current"][cols]
    m = a.merge(b, on=["model_id", "rho_i", "delta", "seed"], suffixes=("_x", "_c"))
    m = m[m["kept_apex_b_x"] & m["kept_apex_b_c"]].copy()
    m["fail"] = ((m["kept_apex_trace_f_x"] - m["kept_apex_trace_f_c"]).abs() > F5_TRACES + TOL) | (
        (m["kept_apex_sample_f_x"] - m["kept_apex_sample_f_c"]).abs() > F5_SAMPLES + TOL)
    r_conds = {(r["model"], r["rho"]) for r in rt if r["R_pair"]}
    g_conds = {(r["model"], r["rho"]) for r in rt if r["G"]}
    key = list(zip(m["model_id"], m["rho_i"]))
    scopes = {
        "judgment_all_pipe_models": [True] * len(m),
        "R_conditions": [k in r_conds for k in key],
        "G_conditions": [k in g_conds for k in key],
        "S4N_only": [k[0].startswith("S4N_") for k in key],
        "S6P_outside_G": [k[0].startswith("S6P_") and k not in g_conds for k in key],
    }
    out = {}
    for name, mask in scopes.items():
        sub = m[pd.Series(mask, index=m.index)]
        out[name] = {"pairs": int(len(sub)), "fail": int(sub["fail"].sum())}
    return out


def original_dt_report(runs: pd.DataFrame) -> dict:
    """REPORT ONLY (F6): auxiliary original-grid runs (dt != 0.098), not judged."""
    o = runs[(runs["dt"].sub(0.098).abs() > 1e-9) & ~runs["ramp"]]
    rep = {"rows": int(len(o))}
    for (m, u), g in o.groupby(["model_id", "rule"]):
        rep[f"{m}|{u}"] = {
            "rows": int(len(g)),
            "accepted_rate": round(float(g["accepted_apex_b"].mean()), 3),
            "n_peaks_median": float(as_float(g["n_peaks"]).median()),
            "cap300_hit_rows": int((g["cap300_hit"] == "True").sum()),
        }
    return rep


def decide(pf1_pass, J, rt) -> dict:
    if not pf1_pass:
        return {"rank": 1, "verdict": "보류(하네스 정합·기록)"}
    s4n_fail = [(m, 20, rate(J, m, 20, "current", 0.0)) for m in PIPE_MODELS
                if m.startswith("S4N_") and rate(J, m, 20, "current", 0.0) < 0.5 - TOL]
    if s4n_fail:
        return {"rank": 2, "verdict": "보류(S4N 수락 단계 성공률 < 0.5)", "detail": s4n_fail}
    if not any(r["Q"] for r in rt):
        return {"rank": 3, "verdict": "보류(자격 Q 없음)"}
    if any(r["R_pair"] for r in rt):
        return {"rank": "4/5", "verdict": "R 성립 — 후보별 F 판정"}
    qgc = [r for r in rt if r["Q"] and r["G"] and r["C"]]
    if qgc:
        if any(r["e"] > RATE_GAP_MAX + TOL for r in qgc):
            return {"rank": 6, "verdict": "약한 재현 — 보류"}
        return {"rank": 7, "verdict": "기작 반증"}
    if any(r["Q"] and r["G"] for r in rt):
        return {"rank": 8, "verdict": "보류(대조 C 불성립)"}
    return {"rank": 9, "verdict": "재현 조건 미달"}


def main() -> None:
    runs, info = load()
    pf1, J, R = pf1_checks(runs, info)
    out = {"note": "batch-1 judgment counts; criteria sealed at bc9a474, decisions D1/D2/P-F1-P-F3 at 7e843d6",
           "P_F1": pf1}
    rt = r_table(J, info, use_cal=False)
    out["R_table"] = rt
    out["R"] = any(r["R_pair"] for r in rt)
    out["decision"] = decide(pf1["pass"], J, rt)
    if out["R"]:
        out["F_by_rule"] = {x: f_checks(J, R, rt, x) for x in CANDIDATES}
        out["verdict_by_rule"] = {x: ("4 고쳐짐(합성)" if v["pass_all"] else "5 불충분") for x, v in out["F_by_rule"].items()}
    out["F5_diagnostic"] = {x: f5_diagnostic(J, info, x) for x in CANDIDATES}
    out["F5_by_scope_report"] = {x: f5_by_scope(J, rt, x) for x in CANDIDATES}
    out["F6_original_dt_report"] = original_dt_report(runs)
    # rates table (accepted stage) for every model, rho, rule, delta
    out["rates"] = {f"{m}|{rho}|{u}": {f"{d:+.2f}": rate(J, m, rho, u, d) for d in DELTAS}
                    for m in PIPE_MODELS for rho in RHOS for u in RULES}
    # F6-type reports
    kept = J[J["model_id"].isin(PIPE_MODELS)].groupby("rule")
    out["F6_kept_rate"] = {u: round(float(g["kept_apex_b"].mean()), 4) for u, g in kept}
    out["F6_lost_reason"] = {u: g["lost_reason"].value_counts().to_dict() for u, g in kept}
    out["F6_cap300_hit_rows"] = {u: int((g["cap300_hit"] == "True").sum()) for u, g in J.groupby("rule")}
    out["F6_F"] = {f"{m}|{u}": round(float(as_float(J[(J["model_id"] == m) & (J["rule"] == u) & (J["delta"] == 0.0)]["F"]).median()), 5)
                   for m in PIPE_MODELS + (NP_MODEL,) for u in RULES}
    out["M0"] = {m: {"Lambda_self": info["models"][m]["harness"]["Lambda"],
                     "Lambda_cal": {k: v / info["models"][m]["harness"]["calibration"]["a0"]
                                    for k, v in info["models"][m]["harness"]["L_noise_free_current"].items()},
                     "argmax_periods_noisy_current": {f"{d:+.2f}": [round(float(x), 3) for x in
                                                     cell(J, m, rho, "current", d)["argmax_level_periods_f"].quantile([0.1, 0.5, 0.9])]
                                                     for rho in (10,) for d in DELTAS},
                     "t_a0_ns": info["models"][m]["harness"]["t_a0_ns"], "t_e_ns": info["models"][m]["harness"]["t_e_ns"],
                     "A0_self": info["models"][m]["harness"]["A0_self"],
                     "A0_cal": info["models"][m]["harness"]["calibration"]["a0"]}
                 for m in S4N_OF}
    # P-F3 sensitivity: sealed literal (Lambda_cal, rho_cal)
    rt_cal = r_table(J, info, use_cal=True)
    out["P_F3"] = {"R_table": rt_cal, "R": any(r["R_pair"] for r in rt_cal),
                   "decision": decide(pf1["pass"], J, rt_cal)}
    if out["P_F3"]["R"]:
        out["P_F3"]["verdict_by_rule"] = {x: ("4" if f_checks(J, R, rt_cal, x)["pass_all"] else "5") for x in CANDIDATES}
    # identity
    sealed = info["seal"]["sealed_files"]
    out["identity"] = {
        "seal_commit": info["seal"]["seal_commit_full"], "seal_commit_time": info["seal"]["seal_commit_time"],
        "sealed_doc_blob": git("rev-parse", f"{SEAL}:{SEALED_DOC}"),
        "interp_doc_blob_at_7e843d6": git("rev-parse", f"{INTERP_COMMIT}:{INTERP_DOC}"),
        "head_commit_run": info["seal"]["head_commit"],
        "harness_started": info["started_at"], "harness_finished": info["finished_at"],
        "runs_csv_sha256_info": info["runs_csv_sha256"],
        "runs_csv_sha256_now": hashlib.sha256((RUN_DIR / "runs.csv").read_bytes()).hexdigest(),
        "pre_run": (RUN_DIR / "pre_run.txt").read_text(encoding="utf-8").strip(),
        "post_run": (RUN_DIR / "post_run.txt").read_text(encoding="utf-8").strip(),
        "sealed_files": {k: v["blob_at_seal"] for k, v in sealed.items()},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    brief = {k: out[k] for k in ("P_F1", "R", "decision")}
    brief["R_table"] = [{k: r[k] for k in ("model", "rho", "delta_e", "rate_e", "rate_l", "e", "Q", "Lambda", "n_floor_held_e", "G", "S4N_gap", "C", "R_pair")} for r in rt]
    brief["P_F3_decision"] = out["P_F3"]["decision"]
    brief["verdict_by_rule"] = out.get("verdict_by_rule")
    print(json.dumps(brief, ensure_ascii=True, indent=1, default=str))


if __name__ == "__main__":
    main()
