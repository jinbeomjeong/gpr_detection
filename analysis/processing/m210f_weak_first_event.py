"""M2-10 (f) weak-first-event lines: read-only aggregation (already-seen 4TU, exploration only).

The 28 lines of spec_depth_v1 section 4-4 F7 supplement group (3): two-event
lines (``air_first_event`` under v2.1) whose first event is not the strongest.
Joins existing outputs only; computes no new zero:
* events / relative amplitudes / single-band reference from
  docs/processing/m29e2_two_event_zero_fourtu.csv (M2-9 v2.1 analysis),
* candidates and apex QC per line from docs/processing/depth_4tu_v2_candidates.csv,
* band-shift QC (sigma_s, flags) from data/processing/m210c_4tu_diag/m210c_4tu_line_diag.csv.
Relative depth shifts with the meta velocity v = c / sqrt(eps_meta):
  zero at the strongest event instead of the first:  -v (t_strong - t_first) / 2
  zero at the same-activity single-band reference:   -v (t_ref - t_first) / 2
(negative = shallower). No truth is used (these lines have none).

Output: docs/processing/m210f_weak_first_event.csv / .json (LF, relative paths)
Run: conda run -n gpr-dev python -I analysis/processing/m210f_weak_first_event.py
"""

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "processing"
SRC_EVENTS = DOC / "m29e2_two_event_zero_fourtu.csv"
SRC_CAND = DOC / "depth_4tu_v2_candidates.csv"
SRC_DIAG = ROOT / "data" / "processing" / "m210c_4tu_diag" / "m210c_4tu_line_diag.csv"
OUT_CSV = DOC / "m210f_weak_first_event.csv"
OUT_JSON = DOC / "m210f_weak_first_event.json"
C = 0.299792458
IDS = {"activity_id": str, "path": str}


def q(s):
    s = pd.Series(s).dropna()
    return [round(float(x), 4) for x in s.quantile([0.1, 0.5, 0.9])] if len(s) else None


def main() -> None:
    ev = pd.read_csv(SRC_EVENTS, dtype=IDS)
    two = ev[ev.n_events >= 2]
    weak = two[two.idx_first != two.idx_strongest].copy()
    cand = pd.read_csv(SRC_CAND, dtype=IDS)
    diag = pd.read_csv(SRC_DIAG, dtype=IDS)
    rows = []
    for r in weak.itertuples():
        t = [float(x) for x in str(r.events_ns).split(";")]
        a = [float(x) for x in str(r.rel_amp).split(";")]
        v = C / math.sqrt(r.eps_r_used)
        i_s = int(r.idx_strongest)
        c = cand[(cand.activity_id == r.activity_id) & (cand.path == r.path)]
        dg = diag[(diag.activity_id == r.activity_id) & (diag.path == r.path)]
        ref = r.single_ref_median_ns
        rows.append({
            "activity_id": r.activity_id, "site": int(r.site), "path": r.path,
            "activity_has_gt": bool(r.activity_has_gt), "dt_ns": r.dt_ns, "eps_r_used": r.eps_r_used,
            "n_events": int(r.n_events), "t_first_ns": t[0], "rel_amp_first": a[0],
            "idx_strongest": i_s, "t_strongest_ns": t[i_s], "gap_strongest_ns": round(t[i_s] - t[0], 3),
            "first_at_record_start": t[0] < 0.5,
            "dd_zero_strongest_m": round(-v * (t[i_s] - t[0]) / 2, 4),
            "single_ref_median_ns": ref, "single_ref_n": r.single_ref_n,
            "first_minus_ref_ns": r.first_minus_ref_ns, "strongest_minus_ref_ns": r.strongest_minus_ref_ns,
            "dd_zero_ref_m": round(-v * (ref - t[0]) / 2, 4) if pd.notna(ref) else None,
            "strongest_matches_ref_0p7ns": bool(pd.notna(r.strongest_minus_ref_ns) and abs(r.strongest_minus_ref_ns) <= 0.7),
            "n_candidates_ok": int((c.status == "ok").sum()),
            # QC verdicts of the ok candidates only, so that in + out + unknown = ok.
            "n_qc_in": int(((c.status == "ok") & (c.qc_verdict == "in")).sum()),
            "n_qc_out": int(((c.status == "ok") & (c.qc_verdict == "out")).sum()),
            "n_qc_unknown": int(((c.status == "ok") & (c.qc_verdict == "unknown")).sum()),
            "median_depth_top_m": round(float(c[c.status == "ok"].depth_top_m.median()), 3) if (c.status == "ok").any() else None,
            "zero_rule_diag": dg.zero_rule.iloc[0] if len(dg) else None,
            "band_shift_std_ns": float(dg.band_shift_std_ns.iloc[0]) if len(dg) else None,
            "zero_shift_large": bool(dg.zero_shift_large.iloc[0]) if len(dg) else None,
            "zero_window_drift": bool(dg.zero_window_drift.iloc[0]) if len(dg) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, lineterminator="\n")
    others = two[two.idx_first == two.idx_strongest]
    out = {
        "note": "already-seen 4TU; read-only joins of existing outputs; exploration only",
        "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SRC_EVENTS, SRC_CAND, SRC_DIAG)},
        "n_two_event_lines": int(len(two)), "n_weak_first": int(len(df)),
        "n_two_event_first_strongest": int(len(others)),
        "by_site": {str(k): int(v) for k, v in df.site.value_counts().sort_index().items()},
        "with_gt": int(df.activity_has_gt.sum()),
        "rel_amp_first_q10_q50_q90": q(df.rel_amp_first),
        "gap_strongest_ns_q10_q50_q90": q(df.gap_strongest_ns),
        "dd_zero_strongest_m_q10_q50_q90": q(df.dd_zero_strongest_m),
        "dd_zero_ref_m_q10_q50_q90": q(df.dd_zero_ref_m),
        "n_with_single_ref": int(df.single_ref_median_ns.notna().sum()),
        "n_strongest_matches_ref": int(df.strongest_matches_ref_0p7ns.sum()),
        "n_first_at_record_start": int(df.first_at_record_start.sum()),
        "candidates_ok": int(df.n_candidates_ok.sum()), "qc_in": int(df.n_qc_in.sum()),
        "qc_out": int(df.n_qc_out.sum()), "qc_unknown": int(df.n_qc_unknown.sum()),
        "band_shift_std_ns_q10_q50_q90": q(df.band_shift_std_ns),
        "n_zero_shift_large": int(df.zero_shift_large.sum()),
        "n_zero_window_drift": int(df.zero_window_drift.sum()),
        "zero_rule_diag": {str(k): int(v) for k, v in df.zero_rule_diag.value_counts().items()},
        "compare_first_strongest_two_event_lines": {
            "n": int(len(others)),
            "band_shift_std_ns_median": None,
        },
    }
    d_oth = diag.merge(others[["activity_id", "path"]], on=["activity_id", "path"])
    out["compare_first_strongest_two_event_lines"]["band_shift_std_ns_median"] = (
        round(float(d_oth.band_shift_std_ns.median()), 4) if len(d_oth) else None)
    s9 = diag[diag.site == 9]
    out["site9_all_lines_band_shift_std_median_by_rule"] = {
        str(k): round(float(v), 4) for k, v in s9.groupby("zero_rule").band_shift_std_ns.median().items()}
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": "docs/processing/m210f_weak_first_event.json", "n": len(df)}))


if __name__ == "__main__":
    main()
