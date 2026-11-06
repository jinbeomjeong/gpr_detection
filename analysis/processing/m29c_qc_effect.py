"""M2-9 C step 2: effect estimate of the locked QC rule B + D on the blinded review sample.

Constants come from docs/processing/m29c_qc_constants.json (written by
m29c_qc_proto.py from synthetic data only, before this script was run). The
review labels (docs/processing/m29c_apex_review.csv, column ``type``) are used
here ONLY to count how many true (T) and false candidates each rule keeps; no
threshold is chosen from them. Ground-truth depth tables and M2-8 matching
results are not read.

Per candidate (same path as eval_depth_4tu, DepthConfig defaults): depth result
with apex_hint, curvature QC (R = 0, air_gap "ground", QC only), record-end flag
and trace-seam flag from the fitting-branch ``filtered`` section of the line.

Outputs: docs/processing/m29c_qc_effect.csv (per candidate),
docs/processing/m29c_qc_effect.json (rule tables).

Run: conda run -n gpr-dev python -I analysis/processing/m29c_qc_effect.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m29c_qc_proto as proto  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = PROJECT / "docs" / "processing"
CONST = DOCS / "m29c_qc_constants.json"
EXPECTED_SHA = "6a88bd8d2501132b0db43eeec2dec9ef347de6e260a9f98aef9a2322ec56ba8c"
G143 = "G143"


def load_constants() -> dict:
    text = CONST.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"constants changed: {sha}")
    const = json.loads(text)
    proto.RECORD_END_PERIODS = const["RECORD_END_PERIODS"]
    proto.SEAM_RATIO = 1.0
    assert proto.SEAM_TIME_PERIODS == const["SEAM_TIME_PERIODS"]
    assert proto.SEAM_AMP_RATIO == const["SEAM_AMP_RATIO"]
    assert proto.SEAM_NOISE_K == const["SEAM_NOISE_K"]
    return const


def rule_table(df: pd.DataFrame, excluded: pd.Series, label: str) -> dict:
    out = {}
    for scope, sel in (
        ("all", np.ones(len(df), bool)),
        ("auto", (df["source"] == "auto_v1").to_numpy()),
        ("manual", (df["source"] == "manual_v0").to_numpy()),
    ):
        sub, exc = df[sel], excluded[sel]
        t = sub["is_true"].to_numpy()
        kept = ~exc.to_numpy()
        n_t, n_f = int(t.sum()), int((~t).sum())
        out[scope] = dict(
            rule=label,
            true_kept=f"{int((t & kept).sum())}/{n_t}",
            false_removed=f"{int((~t & ~kept).sum())}/{n_f}",
            true_frac_before=round(float(t.mean()), 4),
            true_frac_after=round(float(t[kept].mean()), 4) if kept.any() else None,
            n_kept=int(kept.sum()),
        )
    return out


def main() -> None:
    const = load_constants()
    review = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    config = DepthConfig()
    rows = []
    for (act, path), group in review.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg, config)
        for row in group.itertuples():
            rec = dict(blind_id=row.blind_id)
            if pd.isna(row.tau_ns):
                rec.update(verdict="unknown", reason="no_box", record_end=False, seam=False)
                rows.append(rec)
                continue
            trace, sample = round(float(row.trace)), float(row.sample)
            result, qc, d = proto.run_qc(rg, trace, sample, inputs)
            rec.update(
                status=str(result.status), verdict=qc.verdict, reason=qc.reason, r_new=qc.r,
                n_picks_new=qc.n_picks, margin_ns=d["margin_ns"], record_end=d["record_end"],
                seam_score=d["seam_max"], seam_split=d["seam_j"], seam=d["seam"],
                n_samples=rg.n_samples, dt=rg.dt_ns,
            )
            rows.append(rec)
    res = pd.DataFrame(rows)
    df = review.merge(res, on="blind_id", validate="one_to_one")
    df["is_true"] = df["type"] == "T"

    # consistency with the reviewed curvature run (same fit path)
    known = df["r"].notna() & df["r_new"].notna()
    max_r_diff = float((df.loc[known, "r"] - df.loc[known, "r_new"]).abs().max())

    b_old = df["curv_flag"] == "out"  # reviewed: v-bound counted as fit_failed (kept)
    b_new = df["verdict"] == "out"
    d_end = df["record_end"].astype(bool)
    d_seam = df["seam"].astype(bool)
    d_any = d_end | d_seam
    tables = {
        "B_as_reviewed_0.7_1.6": None,
        "B_new": rule_table(df, b_new, "B_new"),
        "D_record_end": rule_table(df, d_end, "D_record_end"),
        "D_seam": rule_table(df, d_seam, "D_seam"),
        "D_any": rule_table(df, d_any, "D_any"),
        "B_new_plus_D": rule_table(df, b_new | d_any, "B_new+D"),
        "E_unknown_excluded_plus_D": rule_table(
            df, b_new | d_any | (df["verdict"] == "unknown"), "E+D (diagnostic)"
        ),
    }
    # B as reviewed with the 0.7-1.6 band (r from the reviewed run, unknown kept)
    r = df["r"]
    b_rev = (df["curv_flag"] != "unknown") & ((r < 0.7) | (r > 1.6))
    tables["B_as_reviewed_0.7_1.6"] = rule_table(df, b_rev, "B_reviewed")
    tables["B_as_reviewed_plus_D"] = rule_table(df, b_rev | d_any, "B_reviewed+D")

    # G143 sensitivity (locked T, re-read W)
    df_w = df.copy()
    df_w.loc[df_w["blind_id"] == G143, "is_true"] = False
    tables["B_new_plus_D_G143_as_W"] = rule_table(df_w, b_new | d_any, "B_new+D, G143=W")

    crosstab = {
        "type_x_verdict": pd.crosstab(df["type"], df["verdict"]).to_dict(),
        "type_x_reason": pd.crosstab(df["type"], df["reason"].fillna("")).to_dict(),
        "type_x_record_end": pd.crosstab(df["type"], d_end).to_dict(),
        "type_x_seam": pd.crosstab(df["type"], d_seam).to_dict(),
        "reason_change_vs_review": pd.crosstab(
            df["unknown_reason"].fillna("").replace("", "(known)"), df["reason"].fillna("")
        ).to_dict(),
    }
    summary = dict(
        constants_sha256=EXPECTED_SHA,
        constants_created=const["created"],
        n=int(len(df)),
        max_abs_r_diff_vs_review=max_r_diff,
        tables=tables,
        crosstab=crosstab,
        flagged_record_end=df.loc[d_end, ["blind_id", "type", "source", "sample", "margin_ns"]]
        .to_dict("records"),
        flagged_seam=df.loc[d_seam, ["blind_id", "type", "source", "seam_score", "seam_split", "trace"]]
        .to_dict("records"),
        b_new_out=df.loc[b_new, ["blind_id", "type", "source", "reason", "r_new"]].to_dict("records"),
    )
    cols = [
        "blind_id", "activity_id", "site", "path", "source", "trace", "sample", "type",
        "confidence", "curv_flag", "unknown_reason", "r", "status", "verdict", "reason",
        "r_new", "margin_ns", "record_end", "seam_score", "seam_split", "seam",
    ]
    df[cols].to_csv(DOCS / "m29c_qc_effect.csv", index=False, float_format="%.4f")
    (DOCS / "m29c_qc_effect.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ["n", "max_abs_r_diff_vs_review"]}))
    for name, tab in tables.items():
        print(name, json.dumps(tab))
    print(json.dumps(crosstab, default=str))
    print("record_end", summary["flagged_record_end"])
    print("seam", summary["flagged_seam"])
    print("b_new_out", summary["b_new_out"])


if __name__ == "__main__":
    main()
