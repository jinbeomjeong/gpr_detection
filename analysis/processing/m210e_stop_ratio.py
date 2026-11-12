"""M2-10 (e) near-apex stop ratio from the committed M2-9 candidate csv (read only).

Recomputes spec_eval_4tu_v1 section 9-8 G on ``docs/processing/depth_4tu_v2_candidates.csv``
(the judged M2-9 run): denominator = ``status == "ok"`` candidates with a QC row,
excluding within-activity copy lines (``duplicate_radargram == "within_activity"``);
a wing stopped within 4 traces when it has a stop reason and ``n_picks <= 3``.
Reports the totals (to compare with the implemented md subsection 538/3,487) and a
per-site split. Already-seen data, diagnostic, not a verdict input.

Output: docs/processing/m210e_stop_ratio.json (LF, relative paths)
Run: conda run -n gpr-dev python -I analysis/processing/m210e_stop_ratio.py
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "processing" / "depth_4tu_v2_candidates.csv"
OUT = ROOT / "docs" / "processing" / "m210e_stop_ratio.json"
STEPS = 4  # PREREG_NEAR_APEX_STEPS in scripts/eval_depth_4tu.py (pre-registered value)


def near(n, stop):
    return isinstance(stop, str) and stop != "" and n <= STEPS - 1


def counts(df):
    n_any = n_both = 0
    reasons = Counter()
    for r in df.itertuples():
        wings = [(r.qc_n_left, r.stop_left), (r.qc_n_right, r.stop_right)]
        hits = [s for n, s in wings if near(n, s)]
        n_any += bool(hits)
        n_both += len(hits) == 2
        reasons.update(hits)
    n = len(df)
    return {
        "n_candidates": n, "n_any": n_any, "pct_any": round(100 * n_any / n, 1) if n else None,
        "n_both": n_both, "n_wings": 2 * n, "n_wings_near": sum(reasons.values()),
        "pct_wings": round(100 * sum(reasons.values()) / (2 * n), 1) if n else None,
        "reasons": dict(sorted(reasons.items())),
    }


def main() -> None:
    c = pd.read_csv(SRC, dtype={"activity_id": str, "path": str, "stop_left": str, "stop_right": str})
    ev = c[(c.status == "ok") & c.qc_n_left.notna() & (c.duplicate_radargram.fillna("") != "within_activity")].copy()
    ev["qc_n_left"] = ev.qc_n_left.astype(int)
    ev["qc_n_right"] = ev.qc_n_right.astype(int)
    out = {
        "note": "already-seen M2-9 candidates (judged run); spec_eval 9-8 G recomputed read-only; diagnostic only",
        "input": "docs/processing/depth_4tu_v2_candidates.csv",
        "input_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "steps": STEPS,
        "total": counts(ev),
        "by_site": {str(s): counts(g) for s, g in ev.groupby("site")},
        "by_qc_verdict": {str(v): counts(g) for v, g in ev.groupby("qc_verdict")},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": "docs/processing/m210e_stop_ratio.json", "total": out["total"]}))


if __name__ == "__main__":
    main()
