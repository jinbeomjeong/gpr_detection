"""M2-9 C: compare the locked blinded re-read (30) with the first review (169).

Reads (after lock only): reread_key.csv, reread_sheet_locked.csv (SHA-256 checked
against reread_lock.json), docs/processing/m29c_apex_review.csv (first read +
curvature flags). No ground-truth depth table and no M2-8 matching result is read.

Outputs:
  docs/processing/m29c_apex_reread.csv       one row per re-read candidate
  docs/processing/m29c_reread_summary.json   agreement, kappa, rule impact

Run: conda run -n gpr-dev python -I analysis/processing/m29c_reread_compare.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT = Path(__file__).resolve().parents[2]
GAL = PROJECT / "data" / "processing" / "m29c_gallery"
DOCS = PROJECT / "docs" / "processing"
TYPES = list("TWLBCXANU")
LEVELS = ["high", "medium", "low"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def rule_masks(d: pd.DataFrame, type_col: str) -> dict[str, pd.Series]:
    known = d["curv_flag"] != "unknown"
    r = d["r"]
    mech = d[type_col].isin(["N", "A"])  # D upper bound: label-based
    a = ~(known & ~r.between(0.7, 1.3))
    b = ~(known & ~r.between(0.7, 1.6))
    return {
        "A_0.7_1.3": a,
        "B_0.7_1.6": b,
        "B+D_label_bound": b & ~mech,
        "E_unknown_drop": known & r.between(0.7, 1.3),
    }


def performance(d: pd.DataFrame, type_col: str) -> list[dict]:
    rows = []
    for scope, sub in (("all", d), ("auto", d[d["source"] == "auto_v1"])):
        t = sub[type_col] == "T"
        for name, keep in rule_masks(sub, type_col).items():
            k = keep
            rows.append(
                {
                    "labels": type_col,
                    "rule": name,
                    "scope": scope,
                    "n_true": int(t.sum()),
                    "n_false": int((~t).sum()),
                    "true_kept": int((k & t).sum()),
                    "false_rejected": int((~k & ~t).sum()),
                    "kept_true_fraction": round(float((k & t).sum() / max(k.sum(), 1)), 4),
                    "precision_before": round(float(t.mean()), 4),
                }
            )
    return rows


def main() -> None:
    lock = json.loads((GAL / "reread_lock.json").read_text(encoding="utf-8"))
    locked = GAL / "reread_sheet_locked.csv"
    digest = hashlib.sha256(locked.read_bytes()).hexdigest().upper()
    assert digest == lock["sha256"], "locked re-read sheet changed after lock"

    key = pd.read_csv(GAL / "reread_key.csv")
    rr = pd.read_csv(locked).rename(
        columns={"type": "reread_type", "confidence": "reread_confidence", "comment": "reread_comment"}
    )
    first = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    m = key.merge(rr, on="reread_id", validate="one_to_one").merge(
        first[
            ["blind_id", "activity_id", "site", "path", "source", "type", "confidence",
             "comment", "curv_flag", "r", "unknown_reason", "pick_n_left", "pick_n_right",
             "d_cand_m"]
        ].rename(columns={"type": "first_type_csv", "confidence": "first_conf_csv",
                          "comment": "first_comment"}),
        on="blind_id",
        validate="one_to_one",
    )
    assert (m["first_type"] == m["first_type_csv"]).all()
    assert (m["first_confidence"] == m["first_conf_csv"]).all()
    m["agree"] = m["first_type"] == m["reread_type"]
    m["T_first"] = m["first_type"] == "T"
    m["T_reread"] = m["reread_type"] == "T"
    m["T_flip"] = m["T_first"] != m["T_reread"]

    n = len(m)
    agree = int(m["agree"].sum())
    kappa9 = cohen_kappa_score(m["first_type"], m["reread_type"], labels=TYPES)
    kappa2 = cohen_kappa_score(m["T_first"], m["T_reread"])
    conf_agree = {}
    for lv in LEVELS:
        s = m[m["first_confidence"] == lv]
        conf_agree[lv] = {"n": int(len(s)), "agree": int(s["agree"].sum()),
                          "wilson95": wilson(int(s["agree"].sum()), len(s))}
    conf_same = int((m["first_confidence"] == m["reread_confidence"]).sum())

    # stratum weights (first-read confidence x type), pool = 168 (G001 excluded)
    pool = first[first["blind_id"] != "G001"]
    cell_n = pool.groupby(["confidence", "type"]).size()
    samp_n = m.groupby(["first_confidence", "first_type"]).size()
    m["weight"] = [cell_n[(c, t)] / samp_n[(c, t)] for c, t in zip(m["first_confidence"], m["first_type"])]
    w_sum = float(m["weight"].sum())
    w_agree = float((m["weight"] * m["agree"]).sum() / w_sum)
    w_tflip = float((m["weight"] * m["T_flip"]).sum() / w_sum)

    confusion = pd.crosstab(m["first_type"], m["reread_type"]).reindex(
        index=TYPES, columns=TYPES, fill_value=0
    )

    # rule impact: substitute re-read labels for the 30 sampled candidates
    full = first.copy()
    full["type_reread_sub"] = full["type"]
    sub_map = dict(zip(m["blind_id"], m["reread_type"]))
    full.loc[full["blind_id"].isin(sub_map), "type_reread_sub"] = full["blind_id"].map(sub_map)
    perf = performance(full, "type") + performance(full, "type_reread_sub")

    out_cols = ["reread_id", "blind_id", "activity_id", "site", "path", "source",
                "first_type", "first_confidence", "first_comment", "reread_type",
                "reread_confidence", "reread_comment", "agree", "T_first", "T_reread",
                "T_flip", "curv_flag", "r", "unknown_reason", "pick_n_left",
                "pick_n_right", "d_cand_m", "weight"]
    m.sort_values("reread_id")[out_cols].to_csv(DOCS / "m29c_apex_reread.csv", index=False)

    summary = {
        "seed_sample": int(key["seed"].iloc[0]),
        "method": key["method"].iloc[0],
        "lock": lock,
        "n": n,
        "agree_9type": agree,
        "agree_9type_wilson95": wilson(agree, n),
        "kappa_9type": round(float(kappa9), 4),
        "agree_T_binary": int((~m["T_flip"]).sum()),
        "kappa_T_binary": round(float(kappa2), 4),
        "agree_by_first_confidence": conf_agree,
        "same_confidence": conf_same,
        "weighted_agree_9type": round(w_agree, 4),
        "weighted_T_flip_rate": round(w_tflip, 4),
        "confusion_first_rows_reread_cols": confusion.to_dict(orient="index"),
        "disagreements": m.loc[~m["agree"], ["reread_id", "blind_id", "first_type",
                                             "first_confidence", "reread_type",
                                             "reread_confidence", "source", "curv_flag",
                                             "r", "unknown_reason"]].to_dict(orient="records"),
        "rule_performance": perf,
    }
    (DOCS / "m29c_reread_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(confusion.to_string())
    print(pd.DataFrame(perf).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ["n", "agree_9type", "agree_9type_wilson95",
                                              "kappa_9type", "agree_T_binary", "kappa_T_binary",
                                              "agree_by_first_confidence", "same_confidence",
                                              "weighted_agree_9type", "weighted_T_flip_rate"]},
                     default=float))


if __name__ == "__main__":
    main()
