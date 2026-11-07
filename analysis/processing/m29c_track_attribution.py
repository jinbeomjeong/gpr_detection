"""M2-9 C follow-up: per-case cause attribution table of the 15 true apexes with unknown curvature.

Diagnosis only. The cause codes below were assigned by inspection of
data/processing/m29c_track_gallery/<id>.png and the replay metrics of
m29c_track_diagnose.py (stop rule, failing time step, window-top flag,
continuation probe, distance of the candidate below the hint-window top,
per-trace envelope peaks near the apex). They are NOT used to set any
constant of a new rule.

Codes
  H  hint-window top truncation: the true ridge lies at / above the hint-window
     top (find_peaks cannot report index 0 or peaks above it), so the earliest
     visible first peak is a locally late trace; its neighbours are earlier.
  J  near-apex pick jitter (flat-topped / doublet envelope, interference):
     an interior reverse step of -0.10 .. -0.25 ns; walking on, the wing goes
     later again (continuation probe >= 7 of 10 steps pass).
  O  true apex outside the +-3-trace hint window: the candidate sits on the
     window's trace edge and the ridge keeps getting earlier beyond it.
  L  envelope-lobe ambiguity: the first-peak candidate is on an earlier, weaker
     hump while the tracker's window argmax takes a later hump.
  I  interference with neighbouring events (layer / crossing capture, trace
     seam, strong shallow layer) that corrupts picks or the fit.
  F  fit stage: enough picks on both sides, but the straight-ray R = 0 fit is
     unstable or on a bound (flat broad top, very shallow pipe).

Output: docs/processing/m29c_track_attribution.csv

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_attribution.py
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / "docs" / "processing"

ATTRIBUTION = {
    "G001": ("H", "O", "ridge traces 198-200 peak at k 137-138, above the hint-window top 140; first visible peak trace 202 (k 140.9) is on the right wing; left step 1 -0.12 ns"),
    "G124": ("H", "J", "traces 583-589 peak at k 160 = hint-window top (index 0, invisible); only trace 587 (k 162) visible; both first steps -0.22/-0.25 ns"),
    "G132": ("H", "L", "hint ~0.8 P late: strong apex humps (k 237-243) above the hint-window top 245; candidate is a weak in-between hump (k 249.7); SNR 8"),
    "G023": ("J", "", "interior reverse -0.137 ns (step 3) and -0.126 ns (step 13); continuation 9/10 pass, net +0.6/+1.3 ns"),
    "G040": ("J", "", "right step 2 interior -0.136 ns; continuation 9/10 pass, net +2.1 ns; left 17 picks"),
    "G153": ("J", "", "left step 3 -0.105 ns, right step 8 -0.134 ns, both interior; continuation 9/10"),
    "G098": ("O", "I", "candidate at hint +3 (edge); right ridge keeps getting earlier (k 181, 180, 180, 178, 178); left wing crossed by clutter"),
    "G131": ("O", "H", "candidate at hint +3 (edge), 1.3 samples below the hint top; traces 445-447 peak earlier (k 128-129): apex further right"),
    "G143": ("O", "", "re-read W: apex ~ +0.20 m right of the candidate; right step 1 at the window top (-0.37 ns), continuation goes earlier (-0.97 ns)"),
    "G017": ("L", "O", "candidate k 132 on an earlier hump; neighbours' window argmax on the later hump (k 142); flat multi-hump envelope (peaks 2 samples apart)"),
    "G026": ("I", "H", "very shallow apex (tau 3.6 ns) right below a strong flat layer; trace seams at +0.2 / +0.7 m; both first steps at the window top"),
    "G081": ("I", "J", "left picks captured by a flat layer from -0.2 m; right interior -0.108 ns at step 9; x0/t0 fit on a bound"),
    "G128": ("I", "", "auto candidate on crossing events, SNR 9; picks jump between events; r 0.45 unstable"),
    "G066": ("F", "", "14 / 12 picks; broad flat top; R = 0 fit unstable (r 1.50)"),
    "G164": ("F", "H", "shallow (0.23 m); 15 / 7 picks; fit on a bound with r 2.06; candidate 1.7 samples below the hint top"),
}

CODE_NAMES = {
    "H": "hint-window top truncation",
    "J": "near-apex jitter",
    "O": "apex outside hint traces",
    "L": "envelope-lobe ambiguity",
    "I": "neighbour interference",
    "F": "fit stage",
}


def main() -> None:
    cases = pd.read_csv(DOCS / "m29c_track_cases.csv", dtype={"activity_id": str})
    proto = pd.read_csv(DOCS / "m29c_track_proto_fourtu.csv")
    one = cases.drop_duplicates("blind_id").set_index("blind_id")
    sides = cases[cases["side"].isin(["left", "right"])].set_index(["blind_id", "side"])
    rows = []
    for bid, (primary, secondary, evidence) in ATTRIBUTION.items():
        o = one.loc[bid]
        p = proto[proto["blind_id"] == bid].iloc[0]
        rec = {
            "blind_id": bid,
            "activity_id": o["activity_id"],
            "site": int(o["site"]),
            "path": o["path"],
            "source": o["source"],
            "hint_trace": int(o["hint_trace"]),
            "apex_trace": int(o["apex_trace"]),
            "d_cand_m": round(float(o["d_cand_m"]), 3),
            "qc_reason": o["qc_reason"],
            "n_left": int(o["n_left"]),
            "n_right": int(o["n_right"]),
            "apex_minus_hint_top_samples": round(float(o["apex_minus_hint_top"]), 2),
            "apex_moved_traces": int(o["apex_moved_traces"]),
        }
        for side in ("left", "right"):
            s = sides.loc[(bid, side)]
            rec[f"{side}_stop"] = s["stop_reason"] + ("_top" if s["stop_reason"] == "reverse" and bool(s["stop_at_top"]) else "")
            rec[f"{side}_stop_step"] = int(s["stop_dist_traces"])
            rec[f"{side}_stop_dtau_ns"] = round(float(s["stop_dtau_ns"]), 3) if pd.notna(s["stop_dtau_ns"]) else None
        rec.update(
            cause_primary=primary,
            cause_secondary=secondary,
            cause_name=CODE_NAMES[primary],
            evidence=evidence,
            v19_T1T2_n_left=int(p["descent+med3_n_left"]),
            v19_T1T2_n_right=int(p["descent+med3_n_right"]),
            v19_T1T2_verdict=p["descent+med3_verdict"],
            v19_T1T2_reason=p["descent+med3_reason"] if isinstance(p["descent+med3_reason"], str) else "",
            figure=f"data/processing/m29c_track_gallery/{bid}.png",
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(DOCS / "m29c_track_attribution.csv", index=False)
    print(out.groupby("cause_primary").size().to_string())
    print(out[["blind_id", "cause_primary", "cause_secondary", "v19_T1T2_verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
