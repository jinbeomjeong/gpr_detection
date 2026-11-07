"""M2-9 C follow-up: prototypes of v1.9 tracking / apex rules (analysis only).

The project code is NOT modified. Inside this process only, three functions
of ``gpr_detection.depth.picks`` are replaced by variants (the default
pipeline calls them by module global name) and ``extract_picks`` is wrapped
so the variants know ``f_c``:

Tracker (``_track_one_side``), stop test ``k < ref - tol``:
  v18       ref = previous pick, tol = 0.1 ns (current spec_depth 5-3).
  step      ref = previous pick, tol = tau_rev.
  runmax    ref = max(apex, previous picks), tol = tau_rev.
  med3      ref = median of the last <= 3 accepted points (apex included),
            tol = tau_rev   (proposed v1.9 rule T2).
  topexit   stop only when the window argmax sits on the window top
            (tol = back window 0.3 ns; sensitivity only).
tau_rev = 2 kappa r_stop / f_c, kappa = 0.258 periods per unit relative
interferer amplitude (m29c_track_interference.py), r_stop = track_stop_rel
0.2 -> 0.1032 / f_c (0.206 ns at 500 MHz, 0.258 ns at 400 MHz).

Hint path (only when ``apex_hint`` is given; the no-hint path is unchanged):
  v18       current 5-7 item 1.
  pad       the hint window top is raised by the tracker back reach
            (track_back_ns), so every peak one admissible tracking step
            above the window is a candidate too (sensitivity only).
  descent   after the hint candidate is chosen, walk to an earlier neighbour
            of the same envelope ridge (local maximum within +-track_back_ns
            of the current time, amplitude >= track_stop_rel x candidate)
            while it is earlier, inside the hint window traces (+-3) and
            the box (proposed v1.9 rule T1).

Parts (CLI ``--part``):
  gprmax  scripts/eval_depth_synthetic.evaluate_all + regression_checks per
          variant -> pass counts, per-model depth / velocity / pick changes.
  fourtu  169 review candidates of m29c_apex_review.csv: picks, QC verdict
          (eval_depth_4tu.curvature_qc), meta-fixed depth per variant.
          Labels are only joined to REPORT the effect; no constant is chosen
          from them. No truth depth table is read (no matching, no MAE).

Outputs: docs/processing/m29c_track_proto_<part>.csv, m29c_track_proto_<part>.json

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_proto.py --part gprmax
     conda run -n gpr-dev python -I analysis/processing/m29c_track_proto.py --part fourtu

Code version: the variants wrap the **v1.8** ``picks.py`` (commit f179e23).
T1 and T2 are in ``picks.py`` since spec_depth v1.9, so this script stops
on the v1.9 code. To reproduce the committed outputs, import the v1.8 package,
e.g. ``git worktree add <dir> f179e23`` and run with ``PYTHONPATH=<dir>\\src``
(without ``-I``, which ignores PYTHONPATH).
"""

import argparse
import dataclasses
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

import eval_depth_4tu as e4  # noqa: E402
import eval_depth_synthetic as es  # noqa: E402

import gpr_detection.depth.estimate as est_mod  # noqa: E402
import gpr_detection.depth.picks as picks_mod  # noqa: E402
from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.picks import ApexCandidate, ApexHint, GridBox  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak  # noqa: E402

V18_COMMIT = "f179e23"
if "track_reverse_ns" not in {f.name for f in dataclasses.fields(DepthConfig)}:
    raise SystemExit(
        "m29c_track_proto.py wraps the v1.8 tracker; this gpr_detection is v1.9 "
        f"(T1/T2 built in). Import the package of commit {V18_COMMIT} (see docstring)."
    )

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = PROJECT / "docs" / "processing"
KAPPA = 0.258
CFG = DepthConfig()
R_STOP = CFG.track_stop_rel
TRACK_BACK_NS = CFG.track_back_ns

ORIGINAL_TRACK = picks_mod._track_one_side
ORIGINAL_EXTRACT = picks_mod.extract_picks
ORIGINAL_FIND = picks_mod.find_apex_candidate
ORIGINAL_HINT_WINDOW = picks_mod._hint_window
STATE = {"f_c_hz": None, "track": "v18", "hint": "v18", "dt_ns": None}

VARIANTS = {
    "v18": ("v18", "v18"),
    "step": ("step", "v18"),
    "runmax": ("runmax", "v18"),
    "med3": ("med3", "v18"),
    "topexit": ("topexit", "v18"),
    "pad": ("v18", "pad"),
    "descent": ("v18", "descent"),
    "descent+med3": ("med3", "descent"),
    "pad+med3": ("med3", "pad"),
}


def period_ns():
    f = STATE["f_c_hz"]
    return 2.0 if f is None else 1e9 / f


def tau_rev_ns():
    return 2.0 * KAPPA * R_STOP * period_ns()


def variant_track(signal, candidate, step, box, limits):
    mode = STATE["track"]
    if mode == "v18":
        return ORIGINAL_TRACK(signal, candidate, step, box, limits)
    dt = TRACK_BACK_NS / limits.back_samples
    tol = tau_rev_ns() / dt
    picks = []
    history = [candidate.k]
    previous_k = candidate.k
    n_samples = signal.shape[0]
    for distance in range(1, limits.max_steps + 1):
        trace = candidate.trace + step * distance
        if not box.i0 <= trace <= box.i1:
            break
        lo = max(0, math.floor(previous_k - limits.back_samples))
        hi = min(n_samples - 1, math.ceil(previous_k + limits.forward_samples))
        if hi <= lo:
            break
        column = signal[:, trace]
        k_peak = lo + int(np.argmax(column[lo : hi + 1]))
        k, amplitude = parabolic_peak(column, k_peak)
        if amplitude < limits.stop_amplitude:
            break
        if mode == "topexit":
            if k_peak == lo and lo > 0:
                break
        else:
            if mode == "step":
                ref = previous_k
            elif mode == "runmax":
                ref = max(history)
            else:  # med3
                ref = float(np.median(history[-3:]))
            if k < ref - tol:
                break
        picks.append((trace, k, amplitude))
        history.append(k)
        previous_k = k
    return picks


def _padded(box, hint, half_traces, half_samples, pad):
    i0 = max(box.i0, hint.trace - half_traces)
    i1 = min(box.i1, hint.trace + half_traces)
    k0 = max(box.k0, math.ceil(hint.sample - half_samples - pad))
    k1 = min(box.k1, math.floor(hint.sample + half_samples))
    if i0 > i1 or k1 <= k0:
        return None
    return GridBox(i0=i0, i1=i1, k0=k0, k1=k1)


def variant_find(signal, box, candidate_rel, *, top_guard, hint=None, hint_halfwidth=None, peak_floor=0.0):
    cand, flags = ORIGINAL_FIND(
        signal, box, candidate_rel, top_guard=top_guard, hint=hint,
        hint_halfwidth=hint_halfwidth, peak_floor=peak_floor,
    )
    if STATE["hint"] != "descent" or hint is None or cand is None or STATE["dt_ns"] is None:
        return cand, flags
    back = TRACK_BACK_NS / STATE["dt_ns"]
    t_lo = max(box.i0, hint.trace - hint_halfwidth[0])
    t_hi = min(box.i1, hint.trace + hint_halfwidth[0])
    floor_amp = R_STOP * cand.amplitude
    n = signal.shape[0]
    current = cand
    while True:
        best = None
        for tr in (current.trace - 1, current.trace + 1):
            if not t_lo <= tr <= t_hi:
                continue
            lo = max(1, math.floor(current.k - back))
            hi = min(n - 2, math.ceil(current.k + back))
            col = signal[:, tr]
            kp = lo + int(np.argmax(col[lo : hi + 1]))
            if kp in (lo, hi):
                continue
            k, a = parabolic_peak(col, kp)
            if a < floor_amp or k >= current.k:
                continue
            if best is None or k < best.k:
                best = ApexCandidate(trace=tr, k=k, amplitude=a)
        if best is None:
            break
        current = best
    if current is not cand:
        flags = (*flags, "apex_descended")
    return current, flags


def wrapped_extract(*args, **kwargs):
    STATE["f_c_hz"] = kwargs.get("f_c_hz")
    STATE["dt_ns"] = kwargs.get("dt_ns")
    return ORIGINAL_EXTRACT(*args, **kwargs)


picks_mod._track_one_side = variant_track
picks_mod.find_apex_candidate = variant_find
picks_mod._hint_window = lambda box, hint, hw: (
    _padded(box, hint, hw[0], hw[1], TRACK_BACK_NS / STATE["dt_ns"])
    if STATE["hint"] == "pad" and STATE["dt_ns"] is not None
    else ORIGINAL_HINT_WINDOW(box, hint, hw)
)
picks_mod.extract_picks = wrapped_extract
est_mod.extract_picks = wrapped_extract
es.extract_picks = wrapped_extract
e4.extract_picks = wrapped_extract


@contextmanager
def variant(name):
    STATE["track"], STATE["hint"] = VARIANTS[name]
    try:
        yield
    finally:
        STATE["track"], STATE["hint"] = "v18", "v18"


def gprmax_part():
    rows, summary = [], {}
    for name in VARIANTS:
        if VARIANTS[name][1] != "v18":
            continue  # hint rules never run on the synthetic regression (no hint)
        with variant(name):
            runs = es.evaluate_all()
            checks = es.regression_checks(runs)
        failed = [f"{c.check_id}/{c.model_id}" for c in checks if not c.passed]
        summary[name] = {"n_checks": len(checks), "n_pass": sum(c.passed for c in checks), "failed": failed}
        for run in runs:
            row = es.model_row(run)
            rows.append(
                {
                    "variant": name,
                    "model_id": row["model_id"],
                    "d_est_m": row["d_est_m"],
                    "d_err_cm": row["d_err_cm"],
                    "v_err_pct": row["v_err_pct"],
                    "radius_free_v_err_pct": row["radius_free_v_err_pct"],
                    "meta_alt_err_cm": row["meta_alt_err_cm"],
                    "eps_qc": row["eps_qc"],
                    "n_picks": row["n_picks"],
                    "picks_n_left": run.picks.n_left,
                    "picks_n_right": run.picks.n_right,
                    "grade": row["grade"],
                    "status": row["status"],
                }
            )
    df = pd.DataFrame(rows).replace("", np.nan)
    base = df[df.variant == "v18"].set_index("model_id")
    for col in ("d_est_m", "v_err_pct", "radius_free_v_err_pct", "meta_alt_err_cm", "eps_qc", "n_picks"):
        df[f"delta_{col}"] = df[col].astype(float).to_numpy() - base.loc[df["model_id"], col].astype(float).to_numpy()
    df.to_csv(DOCS / "m29c_track_proto_gprmax.csv", index=False, float_format="%.5f")
    for name in summary:
        sub = df[df.variant == name]
        summary[name].update(
            max_abs_delta_depth_mm=float(np.nanmax(np.abs(sub["delta_d_est_m"]))) * 1000,
            max_abs_delta_v_err_pct=float(np.nanmax(np.abs(sub["delta_v_err_pct"]))),
            max_abs_delta_radius_free_v_pct=float(np.nanmax(np.abs(sub["delta_radius_free_v_err_pct"]))),
            max_abs_delta_eps_qc=float(np.nanmax(np.abs(sub["delta_eps_qc"]))),
            models_with_pick_change=int((sub["delta_n_picks"].abs() > 0).sum()),
        )
    return summary


def gprmax_hint_part():
    """A6 analogue: the 20 gprMax models with the true apex as hint, every variant."""
    from gpr_detection.io.synthetic import list_synthetic_models, load_synthetic, load_synthetic_truth

    rows = []
    for model_id in list_synthetic_models(es.DEFAULT_SYNTHETIC_ROOT):
        truth = load_synthetic_truth(model_id, root=es.DEFAULT_SYNTHETIC_ROOT)
        rg = load_synthetic(model_id, root=es.DEFAULT_SYNTHETIC_ROOT)
        box = es.supply_box(truth, rg)
        inputs = est_mod.prepare_depth_inputs(rg)
        hint = ApexHint(trace=truth.apex_trace_index, sample=truth.apex_envelope_peak_ns / rg.dt_ns)
        rec = {"model_id": model_id}
        for name in VARIANTS:
            with variant(name):
                res = est_mod.estimate_depth(rg, box, inputs=inputs, apex_hint=hint)
                res_nohint = est_mod.estimate_depth(rg, box, inputs=inputs)
            rec[f"{name}_depth_m"] = res.depth_top_m
            rec[f"{name}_nohint_depth_m"] = res_nohint.depth_top_m
            rec[f"{name}_apex_trace"] = res.apex_trace
            rec[f"{name}_n_picks"] = res.n_picks
            rec[f"{name}_descended"] = "apex_descended" in res.flags
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "m29c_track_proto_gprmax_hint.csv", index=False, float_format="%.6f")
    out = {}
    for name in VARIANTS:
        dd = (df[f"{name}_depth_m"] - df[f"{name}_nohint_depth_m"]).abs()
        dv = (df[f"{name}_depth_m"] - df["v18_depth_m"]).abs()
        out[name] = {
            "max_abs_hint_vs_nohint_mm": float(dd.max() * 1000),
            "max_abs_vs_v18_hint_mm": float(dv.max() * 1000),
            "n_descended": int(df[f"{name}_descended"].sum()),
            "apex_trace_changed_vs_v18": int((df[f"{name}_apex_trace"] != df["v18_apex_trace"]).sum()),
        }
    return out


def fourtu_part():
    review = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    rows = []
    for (act, path), group in review.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = est_mod.prepare_depth_inputs(rg, CFG)
        for row in group.itertuples():
            if pd.isna(row.tau_ns):
                continue
            trace, sample = round(float(row.trace)), float(row.sample)
            box = e4.candidate_box(rg, trace, sample, float(row.tau_ns))
            hint = ApexHint(trace=trace, sample=sample)
            rec = {"blind_id": row.blind_id, "type": row.type, "source": row.source, "site": row.site}
            for name in VARIANTS:
                with variant(name):
                    result = est_mod.estimate_depth(rg, box, CFG, inputs=inputs, apex_hint=hint)
                    qc = e4.curvature_qc(rg, box, inputs, hint, result, CFG)
                rec[f"{name}_n_left"] = qc.n_left
                rec[f"{name}_n_right"] = qc.n_right
                rec[f"{name}_verdict"] = str(qc.verdict)
                rec[f"{name}_reason"] = qc.reason
                rec[f"{name}_r"] = qc.r
                rec[f"{name}_depth_m"] = result.depth_top_m
                rec[f"{name}_apex_trace"] = result.apex_trace
                rec[f"{name}_apex_sample"] = result.apex_sample
                rec[f"{name}_descended"] = "apex_descended" in result.flags
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "m29c_track_proto_fourtu.csv", index=False, float_format="%.5f")
    is_t = df["type"] == "T"
    auto = df["source"] == "auto_v1"
    summary = {}
    for name in VARIANTS:
        v = df[f"{name}_verdict"]
        d = (df[f"{name}_depth_m"] - df["v18_depth_m"]).abs()
        summary[name] = {
            "T": v[is_t].value_counts().to_dict(),
            "false": v[~is_t].value_counts().to_dict(),
            "T_unknown_reasons": df.loc[is_t & (v == "unknown"), f"{name}_reason"].value_counts().to_dict(),
            "auto_T": v[is_t & auto].value_counts().to_dict(),
            "auto_false": v[~is_t & auto].value_counts().to_dict(),
            "kept_T_rule_BD_excl_unknown(E)": int(((v == "in") & is_t).sum()),
            "kept_false_E": int(((v == "in") & ~is_t).sum()),
            "kept_auto_T_E": int(((v == "in") & is_t & auto).sum()),
            "kept_auto_false_E": int(((v == "in") & ~is_t & auto).sum()),
            "depth_change_vs_v18_m_q50_q90_max": [float(d.quantile(0.5)), float(d.quantile(0.9)), float(d.max())],
            "n_depth_changed_gt_1mm": int((d > 0.001).sum()),
            "n_depth_changed_gt_1cm": int((d > 0.01).sum()),
            "n_descended": int(df[f"{name}_descended"].sum()),
            "transitions_T": pd.crosstab(df.loc[is_t, "v18_verdict"], v[is_t]).to_dict("index"),
            "transitions_false": pd.crosstab(df.loc[~is_t, "v18_verdict"], v[~is_t]).to_dict("index"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["gprmax", "gprmax_hint", "fourtu"], required=True)
    args = parser.parse_args()
    out = {"tau_rev_ns_500MHz": 2 * KAPPA * R_STOP * 2.0, "tau_rev_ns_400MHz": 2 * KAPPA * R_STOP * 2.5}
    parts = {"gprmax": gprmax_part, "gprmax_hint": gprmax_hint_part, "fourtu": fourtu_part}
    out[args.part] = parts[args.part]()
    (DOCS / f"m29c_track_proto_{args.part}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
