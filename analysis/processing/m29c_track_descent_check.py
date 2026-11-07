"""M2-9 C follow-up: v1.9 implementation checks (analysis only, v1.9 package).

Parts (CLI ``--part``, default all):

  fourtu  The 169 review candidates of m29c_apex_review.csv through the v1.9
          eval path (``eval_depth_4tu._run_candidate``, QC on). Per candidate:
          was the apex descended (rule T1), where did the descent end relative
          to the search top ``search_k0`` and the mute line ``t_z + 1/f_c``,
          how far did it move (traces, ns), wing stop reasons, QC B verdict and
          rule D flags. ``descend_apex`` is wrapped in-process (recording only)
          to get the pre-descent candidate. Review labels are joined ONLY to
          count true (T) and false candidates for the E-decision table; no
          constant is chosen from them. No truth depth table, no M2-8 matching.
  fourtu_guard  Sensitivity only (NOT the v1.9 rule): the same run with the
          descent steps also obeying the apex-search top guard (integer ridge
          peak index must be > k_top + top_guard_samples); lists the
          candidates whose apex, descent or QC verdict changes.
  gprmax  The 20 gprMax models with the true apex as hint (A6 analogue) and
          without hint: descents, descents above search top / mute, depth
          difference hint vs no hint.
  ctests  Mechanism of the changed QC test expectations C2 (flat layer 0.5),
          C3 (X crossing) and C5 (record end): pick counts, stop reasons and
          outward wing slopes under v1.9 (synthetic generators of
          tests/test_eval_depth_4tu.py, imported, not modified).

Outputs (docs/processing/):
  m29c_track_descent_check_fourtu.csv, m29c_track_descent_check_fourtu_guard.csv,
  m29c_track_descent_check_gprmax.csv, m29c_track_descent_check_ctests.csv,
  m29c_track_descent_check.json

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_descent_check.py
Single process, about 2 min. Deterministic (no random numbers; the C-test
generators use their fixed noise seed 0).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "tests"))

import eval_depth_4tu as e4  # noqa: E402
import eval_depth_synthetic as es  # noqa: E402

import gpr_detection.depth.estimate as est_mod  # noqa: E402
import gpr_detection.depth.picks as picks_mod  # noqa: E402
from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.picks import ApexHint, mute_sample  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402

if "track_reverse_kappa" not in {f.name for f in dataclasses.fields(DepthConfig)}:
    raise SystemExit("this check needs the v1.9 gpr_detection package (T1/T2 built in)")

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = PROJECT / "docs" / "processing"
CFG = DepthConfig()

ORIGINAL_DESCEND = picks_mod.descend_apex
ORIGINAL_E4_EXTRACT = e4.extract_picks
DESCENT_LOG: list[dict] = []
PICKS_LOG: list = []


ORIGINAL_RIDGE = picks_mod._earlier_ridge_peak
GUARD = {"on": False, "k_min_int": None}


def guarded_ridge(column, current_k, back_samples):
    """Sensitivity variant only: reject ridge peaks in the top-guard band too.

    The apex search rejects integer peak indices ``k_p - k_top <= top_guard``
    (spec_depth 5-7 item 2); this applies the same test to descent steps.
    """
    peak = ORIGINAL_RIDGE(column, current_k, back_samples)
    if peak is None or not GUARD["on"]:
        return peak
    return None if round(peak[0]) < GUARD["k_min_int"] else peak


def recording_descend(signal, candidate, box, hint, hint_traces, back_samples, floor_amplitude):
    GUARD["k_min_int"] = box.k0 + CFG.top_guard_samples + 1
    out = ORIGINAL_DESCEND(signal, candidate, box, hint, hint_traces, back_samples, floor_amplitude)
    DESCENT_LOG.append(
        dict(pre_trace_local=candidate.trace, pre_k=candidate.k, out_trace_local=out.trace,
             out_k=out.k, hint_trace_local=hint.trace, hint_sample=hint.sample,
             n_rows=signal.shape[0])
    )
    return out


def recording_extract(*args, **kwargs):
    picks = ORIGINAL_E4_EXTRACT(*args, **kwargs)
    PICKS_LOG.append(picks)
    return picks


picks_mod.descend_apex = recording_descend
picks_mod._earlier_ridge_peak = guarded_ridge
e4.extract_picks = recording_extract


def period_ns(f_c_hz):
    return 2.0 if f_c_hz is None else 1e9 / f_c_hz


def wing_slopes(picks, dt_ns, dx_m):
    """Outward slope dt/d|x| (ns/m) of each wing by least squares; NaN if < 2 picks."""
    if picks is None or picks.apex is None or picks.n_picks == 0:
        return math.nan, math.nan
    tr, k = picks.trace_idx, picks.k_pick
    out = []
    for side in (-1, 1):
        sel = (tr - picks.apex.trace) * side >= 0
        if sel.sum() < 3:
            out.append(math.nan)
            continue
        dist = np.abs(tr[sel] - picks.apex.trace) * dx_m
        out.append(float(np.polyfit(dist, k[sel] * dt_ns, 1)[0]))
    return out[0], out[1]


# ----------------------------------------------------------------------------- 4TU


def fourtu_guard_part():
    """Sensitivity: descent steps also obey the top guard (not the v1.9 rule)."""
    base = pd.read_csv(DOCS / "m29c_track_descent_check_fourtu.csv")
    GUARD["on"] = True
    try:
        df, _ = fourtu_part(csv_name="m29c_track_descent_check_fourtu_guard.csv")
    finally:
        GUARD["on"] = False
    df = pd.read_csv(DOCS / "m29c_track_descent_check_fourtu_guard.csv")  # same dtypes as base
    m = base.merge(df, on="blind_id", suffixes=("", "_g"))
    for col in ("qc_reason", "qc_reason_g"):
        m[col] = m[col].fillna("")
    changed = (
        (m["apex_sample"] - m["apex_sample_g"]).abs().gt(1e-9)
        | (m["qc_verdict"] != m["qc_verdict_g"])
        | (m["descended"] != m["descended_g"])
    )
    cols = ["blind_id", "type", "source", "apex_sample", "apex_sample_g", "descended",
            "descended_g", "qc_verdict", "qc_verdict_g", "qc_reason", "qc_reason_g",
            "depth_top_m", "depth_top_m_g"]
    return dict(n_changed=int(changed.sum()), rows=m.loc[changed, cols].round(4).to_dict("records"))


def fourtu_part(csv_name="m29c_track_descent_check_fourtu.csv"):
    review = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    rows = []
    for (act, path), group in review.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = est_mod.prepare_depth_inputs(rg, CFG)
        seam_line = e4._line_seam_scores(rg, inputs)
        p_ns = period_ns(rg.f_c_hz)
        t_zero = inputs.zero.t_zero_ns
        k_mute = None if t_zero is None else mute_sample(t_zero, p_ns, rg.dt_ns)
        for row in group.itertuples(index=False):
            DESCENT_LOG.clear()
            PICKS_LOG.clear()
            cand = e4._run_candidate(rg, inputs, row, seam_line, True)
            res, qc = cand.result, cand.qc
            curv = qc.curvature
            rec = dict(
                blind_id=row.blind_id, activity_id=act, path=path, site=row.site,
                source=row.source, type=row.type, dt_ns=rg.dt_ns, status=str(res.status),
                hint_trace=cand.candidate_trace, hint_sample=cand.candidate_sample,
                box_k0=cand.box_k0, search_k0=cand.search_k0, k_mute=k_mute,
                apex_trace=res.apex_trace, apex_sample=res.apex_sample,
                descended="apex_descended" in res.flags, depth_top_m=res.depth_top_m,
                qc_verdict=str(curv.verdict), qc_reason=curv.reason, qc_r=curv.r,
                n_left=curv.n_left, n_right=curv.n_right,
                stop_left="" if curv.stop_left is None else curv.stop_left.value,
                stop_right="" if curv.stop_right is None else curv.stop_right.value,
                record_end=qc.at_record_end, seam=qc.on_trace_seam, qc_excluded=qc.excluded_bd,
            )
            if cand.search_k0 is not None:
                half = CFG.hint_periods * p_ns / rg.dt_ns
                rec["hint_top"] = max(cand.search_k0, math.ceil(cand.candidate_sample - half))
            # first descend call = depth path (QC re-call is identical)
            if DESCENT_LOG and rec["descended"]:
                d = DESCENT_LOG[0]
                rec["pre_k"] = d["pre_k"]
                rec["descent_traces"] = d["out_trace_local"] - d["pre_trace_local"]
                rec["descent_ns"] = (d["pre_k"] - d["out_k"]) * rg.dt_ns
                rec["apex_minus_hint_trace"] = d["out_trace_local"] - d["hint_trace_local"]
                assert abs(d["out_k"] - res.apex_sample) < 1e-9
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["above_search_k0"] = df["descended"] & (df["apex_sample"] < df["search_k0"])
    df["above_mute"] = df["descended"] & (df["apex_sample"] < df["k_mute"])
    df["above_hint_top"] = df["descended"] & (df["apex_sample"] < df["hint_top"])
    df["apex_minus_mute_samples"] = df["apex_sample"] - df["k_mute"]
    df["apex_minus_search_k0"] = df["apex_sample"] - df["search_k0"]
    df.to_csv(DOCS / csv_name, index=False, float_format="%.5f")

    # consistency with the committed prototype (descent+med3)
    proto = pd.read_csv(DOCS / "m29c_track_proto_fourtu.csv")
    m = df.merge(proto, on="blind_id", suffixes=("", "_p"))
    consistency = dict(
        n_joined=int(len(m)),
        verdict_equal=int((m["qc_verdict"] == m["descent+med3_verdict"]).sum()),
        descended_equal=int((m["descended"] == m["descent+med3_descended"]).sum()),
        max_abs_depth_diff_m=float((m["depth_top_m"] - m["descent+med3_depth_m"]).abs().max()),
        max_abs_apex_sample_diff=float((m["apex_sample"] - m["descent+med3_apex_sample"]).abs().max()),
    )

    desc = df[df["descended"]]
    descent = dict(
        n_candidates=int(len(df)),
        n_descended=int(len(desc)),
        n_descended_by_type={k: int(v) for k, v in desc["type"].value_counts().items()},
        n_descended_auto=int((desc["source"] == "auto_v1").sum()),
        n_above_search_k0=int(df["above_search_k0"].sum()),
        n_above_mute=int(df["above_mute"].sum()),
        n_above_hint_top=int(df["above_hint_top"].sum()),
        n_search_k0_is_mute=int((df["search_k0"] == df["k_mute"]).sum()),
        descent_ns_max=float(desc["descent_ns"].max()) if len(desc) else None,
        descent_ns_values=sorted(round(float(x), 4) for x in desc["descent_ns"]),
        descent_traces_values=sorted(int(x) for x in desc["descent_traces"]),
        n_stop_at_hint_edge=int((desc["apex_minus_hint_trace"].abs() == CFG.hint_traces).sum()),
        min_apex_minus_mute_samples_descended=float(desc["apex_minus_mute_samples"].min())
        if len(desc) else None,
        min_apex_minus_search_k0_descended=float(desc["apex_minus_search_k0"].min())
        if len(desc) else None,
        descended_rows=desc[
            ["blind_id", "type", "source", "dt_ns", "hint_trace", "apex_trace", "hint_sample",
             "pre_k", "apex_sample", "hint_top", "search_k0", "k_mute", "box_k0",
             "descent_traces", "descent_ns", "apex_minus_hint_trace", "qc_verdict", "qc_reason"]
        ].round(3).to_dict("records"),
    )

    has_picks = df["stop_left"] != ""
    stops = dict(
        n_tracked=int(has_picks.sum()),
        left={k: int(v) for k, v in df.loc[has_picks, "stop_left"].value_counts().items()},
        right={k: int(v) for k, v in df.loc[has_picks, "stop_right"].value_counts().items()},
    )
    short = has_picks & (df[["n_left", "n_right"]].min(axis=1) <= 3)
    is_t = df["type"] == "T"
    stops["one_wing_within_4_traces"] = dict(
        T=f"{int((short & is_t).sum())}/{int((has_picks & is_t).sum())}",
        false=f"{int((short & ~is_t).sum())}/{int((has_picks & ~is_t).sum())}",
    )

    verdicts = dict(
        T={k: int(v) for k, v in df.loc[is_t, "qc_verdict"].value_counts().items()},
        false={k: int(v) for k, v in df.loc[~is_t, "qc_verdict"].value_counts().items()},
        T_unknown_reasons={k: int(v) for k, v in df.loc[is_t & (df["qc_verdict"] == "unknown"), "qc_reason"].value_counts().items()},
        false_unknown_reasons={k: int(v) for k, v in df.loc[~is_t & (df["qc_verdict"] == "unknown"), "qc_reason"].value_counts().items()},
    )
    return df, dict(consistency=consistency, descent=descent, stops=stops, verdicts=verdicts,
                    e_table=e_table(df))


def _scope_counts(df, kept, scope):
    sel = np.ones(len(df), bool) if scope == "all" else (df["source"] == scope).to_numpy()
    t = (df["type"] == "T").to_numpy()[sel]
    k = kept.to_numpy()[sel]
    n_t, n_f = int(t.sum()), int((~t).sum())
    kt, kf = int((t & k).sum()), int((~t & k).sum())
    n_kept = kt + kf
    if n_kept:
        ci = binomtest(kt, n_kept).proportion_ci(confidence_level=0.95, method="exact")
        prec = [round(kt / n_kept, 4), round(ci.low, 4), round(ci.high, 4)]
    else:
        prec = [None, None, None]
    return dict(T_kept=f"{kt}/{n_t}", false_kept=f"{kf}/{n_f}", false_removed=f"{n_f - kf}/{n_f}",
                n_kept=n_kept, precision_ci95=prec)


def e_table(df):
    out_b = df["qc_verdict"] == "out"
    unk = df["qc_verdict"] == "unknown"
    d_any = df["record_end"].astype(bool) | df["seam"].astype(bool)
    ok = df["status"] == "ok"
    rules = {
        "none": pd.Series(False, index=df.index),
        "B": out_b,
        "B+D": out_b | d_any,
        "B+E": out_b | unk,
        "B+D+E": out_b | d_any | unk,
    }
    table = {}
    for name, excluded in rules.items():
        kept = ~excluded
        table[name] = {s: _scope_counts(df, kept, s) for s in ("all", "auto_v1", "manual_v0")}
        table[name + " (ok only)"] = {
            s: _scope_counts(df[ok].reset_index(drop=True), kept[ok].reset_index(drop=True), s)
            for s in ("all", "auto_v1", "manual_v0")
        }
    # what E removes on top of B+D, by type and reason
    e_only = unk & ~(out_b | d_any)
    table["E_removes_on_top_of_BD"] = dict(
        by_type={k: int(v) for k, v in df.loc[e_only, "type"].value_counts().items()},
        by_reason_T={k: int(v) for k, v in df.loc[e_only & (df["type"] == "T"), "qc_reason"].value_counts().items()},
        by_reason_false={k: int(v) for k, v in df.loc[e_only & (df["type"] != "T"), "qc_reason"].value_counts().items()},
        T_blind_ids=sorted(df.loc[e_only & (df["type"] == "T"), "blind_id"]),
        auto_T_blind_ids=sorted(df.loc[e_only & (df["type"] == "T") & (df["source"] == "auto_v1"), "blind_id"]),
    )
    return table


# --------------------------------------------------------------------------- gprMax


def gprmax_part():
    from gpr_detection.io.synthetic import list_synthetic_models, load_synthetic, load_synthetic_truth

    rows = []
    for model_id in list_synthetic_models(es.DEFAULT_SYNTHETIC_ROOT):
        truth = load_synthetic_truth(model_id, root=es.DEFAULT_SYNTHETIC_ROOT)
        rg = load_synthetic(model_id, root=es.DEFAULT_SYNTHETIC_ROOT)
        box = es.supply_box(truth, rg)
        inputs = est_mod.prepare_depth_inputs(rg)
        hint = ApexHint(trace=truth.apex_trace_index, sample=truth.apex_envelope_peak_ns / rg.dt_ns)
        res = est_mod.estimate_depth(rg, box, inputs=inputs, apex_hint=hint)
        res0 = est_mod.estimate_depth(rg, box, inputs=inputs)
        p_ns = period_ns(rg.f_c_hz)
        t_zero = inputs.zero.t_zero_ns
        k_mute = None if t_zero is None else mute_sample(t_zero, p_ns, rg.dt_ns)
        search_k0 = box.k0 if k_mute is None else max(box.k0, k_mute)
        rows.append(dict(
            model_id=model_id, descended="apex_descended" in res.flags,
            apex_sample=res.apex_sample, search_k0=search_k0, k_mute=k_mute,
            apex_trace=res.apex_trace, apex_trace_nohint=res0.apex_trace,
            depth_hint_m=res.depth_top_m, depth_nohint_m=res0.depth_top_m,
            n_picks_hint=res.n_picks, n_picks_nohint=res0.n_picks,
        ))
    df = pd.DataFrame(rows)
    df["above_search_k0"] = df["descended"] & (df["apex_sample"] < df["search_k0"])
    df["above_mute"] = df["descended"] & (df["apex_sample"] < df["k_mute"])
    df.to_csv(DOCS / "m29c_track_descent_check_gprmax.csv", index=False, float_format="%.6f")
    dd = (df["depth_hint_m"] - df["depth_nohint_m"]).abs()
    return dict(
        n_models=int(len(df)), n_descended=int(df["descended"].sum()),
        n_above_search_k0=int(df["above_search_k0"].sum()), n_above_mute=int(df["above_mute"].sum()),
        max_abs_depth_hint_vs_nohint_mm=float(dd.max() * 1000),
        apex_trace_changed=int((df["apex_trace"] != df["apex_trace_nohint"]).sum()),
    )


# ---------------------------------------------------------------------------- C tests


def ctests_part():
    import test_eval_depth_4tu as t  # tests module: generators only, nothing is modified

    cases = []
    x_m, _, inside = t._x_and_mask(1.0)
    for amp in (0.5, 2.0):
        cases.append((f"C2 flat layer amp {amp}",
                      t._layer_line(np.full(len(x_m), t.LAYER_TIME_NS), amp * inside),
                      t.CENTRE_TRACE, (t.T_SRC_NS + t.LAYER_TIME_NS) / t.QC_DT_NS))
    for slope in (5.0, 10.0, 15.0, 20.0, 30.0):
        cases.append((f"C3 X {slope:g} ns/m", t._crossing_line(slope, cross=True),
                      t.CENTRE_TRACE, (t.T_SRC_NS + t.CROSSING_TIME_NS) / t.QC_DT_NS))
    for margin in (0.8, 1.5, 3.0, 6.0):
        spec = t._record_end_spec(margin)
        cases.append((f"C5 record end {margin:g} ns", t._radargram(spec), *t._true_apex(spec)))
    rows = []
    for name, rg, trace, sample in cases:
        PICKS_LOG.clear()
        (cand,) = e4.run_line(rg, t._candidate_table(rg, trace, sample), apex_qc=True)
        picks = PICKS_LOG[-1] if PICKS_LOG else None
        curv = cand.qc.curvature
        sl, sr = wing_slopes(picks, rg.dt_ns, rg.dx_m)
        rows.append(dict(
            case=name, verdict=str(curv.verdict), reason=curv.reason, r=curv.r,
            inlier_frac=curv.inlier_frac, n_left=curv.n_left, n_right=curv.n_right,
            stop_left="" if curv.stop_left is None else curv.stop_left.value,
            stop_right="" if curv.stop_right is None else curv.stop_right.value,
            slope_left_ns_per_m=sl, slope_right_ns_per_m=sr,
            descended="apex_descended" in cand.result.flags,
            record_end=cand.qc.at_record_end, excluded=cand.qc.excluded_bd,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "m29c_track_descent_check_ctests.csv", index=False, float_format="%.4f")
    tau = picks_mod.reverse_tolerance_ns(500e6, CFG)
    return dict(
        tau_rev_ns_500MHz=tau,
        sustained_rise_threshold_ns_per_trace_v19=tau / 2,
        sustained_rise_threshold_ns_per_m_dx002_v19=tau / 2 / 0.02,
        sustained_rise_threshold_ns_per_m_dx002_v18=0.1 / 0.02,
        rows=df.to_dict("records"),
    )


def _nonfinite_to_none(value: object) -> object:
    """Replace NaN/±Inf floats (also NumPy scalars) with None, recursing containers.

    Standard JSON has no NaN/Infinity tokens, so they are written as null.
    """
    if isinstance(value, dict):
        return {key: _nonfinite_to_none(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_nonfinite_to_none(item) for item in value]
    if isinstance(value, float | np.floating) and not math.isfinite(value):
        return None
    return value


def _to_strict_json(data: object) -> str:
    return json.dumps(_nonfinite_to_none(data), indent=1, default=str, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", choices=["all", "fourtu", "fourtu_guard", "gprmax", "ctests"], default="all"
    )
    args = parser.parse_args()
    out_path = DOCS / "m29c_track_descent_check.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if args.part in ("all", "fourtu"):
        _, out["fourtu"] = fourtu_part()
    if args.part in ("all", "fourtu_guard"):
        out["fourtu_guard"] = fourtu_guard_part()
    if args.part in ("all", "gprmax"):
        out["gprmax"] = gprmax_part()
    if args.part in ("all", "ctests"):
        out["ctests"] = ctests_part()
    out_text = _to_strict_json(out)
    out_path.write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
