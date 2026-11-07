"""M2-9 two-event zero rule: prototype of candidate rules (analysis only).

The project code is NOT modified. Inside this process only,
``gpr_detection.preprocess.timezero._air_zero`` (called by module global name
from ``_choose_rule``) is replaced by a variant dispatcher. Variants for an
air-coupled mean trace with two or more early events (one event: unchanged):

  current    last event in the window (spec_depth v2.0 ``air_surface_late``).
  first      earliest event (merged band; later events = sub-surface).
  strongest  event with the largest envelope amplitude.
  first_h    proposed v2.1 rule: if the antenna height ``h`` is known, the
             event whose gap to the first event matches
             ``gap_h = (2 sqrt(h^2 + s^2/4) - s) / c`` within ``0.5 / f_c``
             is the surface reflection; if ``gap_h < 0.75 / f_c`` (direct and
             surface merge) or nothing matches, the first event. ``h`` unknown
             -> first event.

``first`` / ``strongest`` / ``first_h`` without a match return the rule
``air_single_event`` here (prototype shortcut: label ``merged``, sigma 0.25,
no refraction QC 2a); the spec gives them their own name ``air_first_event``.

Parts (CLI ``--part``):
  analytic  analytic air-coupled B-scans (tests/synthetic_bscan.py, seed 0):
            direct + surface (D+S, h 0.25-0.5 m), merged band + sub-surface
            layer (B+L, h 0.05 m), D+S + layer. Truth zero = geometric surface
            reflection time ``t_src + 2 sqrt(h^2 + s^2/4) / c``.
  gprmax    scripts/eval_depth_synthetic.evaluate_all + regression_checks per
            variant (h known = json ``air_gap_m``; R1-R13 129 checks).
  fourtu    every line of the activities that contain an ``air_surface_late``
            line in docs/processing/m29e_zero_survey_lines.csv: events, zero per
            variant, implied depth change ``v_meta (t_cur - t_new) / 2``,
            comparison with the same-activity single-band time, per-group
            kinematics (11-trace stacks) and relative polarity. No ground-truth
            depth is read; ``activity_has_gt`` (membership only) is copied from
            the survey csv.

Outputs (docs/processing/): m29e2_two_event_zero_<part>.csv and
m29e2_two_event_zero_<part>.json.

Run (single process, BelowNormal, BLAS 1 thread):
    conda run -n gpr-dev python -I analysis/processing/m29e2_two_event_zero.py --part analytic
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "tests"))

import gpr_detection.preprocess.timezero as tz  # noqa: E402
from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG  # noqa: E402
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS as C  # noqa: E402
from gpr_detection.preprocess.filters import (  # noqa: E402
    period_ns_or_default,
    prepare_fitting_branch,
)
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

DOCS = PROJECT / "docs" / "processing"
SURVEY_CSV = DOCS / "m29e_zero_survey_lines.csv"
VARIANTS = ("current", "first", "strongest", "first_h")
MATCH_PERIODS = 0.5
"""Known-h match tolerance in periods (0.5 / f_c, 1.0 ns at 500 MHz)."""
GROUP_TRACES = 11
TRACK_HALF_NS = 0.5

ORIGINAL_AIR_ZERO = tz._air_zero
STATE: dict[str, object] = {"variant": "current", "h_of": lambda radargram: None, "log": []}


# --------------------------------------------------------------- the variants


def gap_for_height_ns(h_m: float, s_m: float | None) -> float:
    s = 0.0 if s_m is None else s_m
    return (2.0 * math.hypot(h_m, s / 2.0) - s) / C


def choose_index(variant: str, events, radargram, h_m: float | None) -> tuple[int, str]:
    """Index of the zero event and the reason, for >= 2 events."""
    k = np.asarray(events.events_k)
    if variant == "current":
        return len(k) - 1, "last"
    if variant == "first":
        return 0, "first"
    if variant == "strongest":
        return int(np.argmax(events.amplitudes)), "strongest"
    if variant != "first_h":
        raise ValueError(variant)
    if h_m is None or events.dt_ns is None:
        return 0, "h_unknown_first"
    period = period_ns_or_default(radargram.f_c_hz, tz.DEFAULT_PERIOD_NS)
    gap_h = gap_for_height_ns(h_m, radargram.antenna_offset_m)
    if gap_h < tz.SEPARATION_PERIODS * period:
        return 0, "h_merged_first"
    gaps = (k[1:] - k[0]) * events.dt_ns
    j = int(np.argmin(np.abs(gaps - gap_h)))
    if abs(gaps[j] - gap_h) <= MATCH_PERIODS * period:
        return j + 1, "h_surface_matched"
    return 0, "h_surface_not_found_first"


def air_zero_variant(radargram, events):
    variant = str(STATE["variant"])
    if events.n_events < 2:
        return ORIGINAL_AIR_ZERO(radargram, events)
    h_m = STATE["h_of"](radargram)
    index, reason = choose_index(variant, events, radargram, h_m)
    STATE["log"].append(
        {"line_id": radargram.line_id, "variant": variant, "index": index, "reason": reason}
    )
    k_event = events.events_k[index]
    if reason in ("last", "h_surface_matched"):
        note = "air_surface_late (prototype)"
        return tz._RuleChoice(
            tz.ZeroRule.AIR_SURFACE_LATE, k_event, k_event, "surface_reflection", (note,)
        )
    note = f"air_first_event (prototype, {reason})"
    return tz._RuleChoice(tz.ZeroRule.AIR_SINGLE_EVENT, k_event, k_event, "merged", (note,))


tz._air_zero = air_zero_variant


class use_variant:
    def __init__(self, variant: str, h_of=None):
        self.variant, self.h_of = variant, h_of

    def __enter__(self):
        STATE["variant"] = self.variant
        if self.h_of is not None:
            STATE["h_of"] = self.h_of
        return self

    def __exit__(self, *exc):
        STATE["variant"] = "current"
        STATE["h_of"] = lambda radargram: None


# --------------------------------------------------------------- analytic part


def analytic_part() -> dict[str, object]:
    from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram

    eps, f_c, s = 9.0, 500e6, 0.1
    v = C / math.sqrt(eps)
    pipe_depth = 1.5

    def spec_for(h, direct, surface, layers):
        spec = BScanSpec(
            eps_r=eps, f_c_hz=f_c, air_gap_m=h, offset_m=s, d_top_m=pipe_depth,
            n_traces=61, n_samples=900, direct_amplitude=direct, surface_amplitude=surface,
        )
        soil = cylinder_times(spec.x_m, spec.x0_m, pipe_depth, v, 0.0)
        spec.pipe_times_ns = soil + 2.0 * h / C
        surface_ns = 2.0 * math.hypot(h, s / 2.0) / C
        ones = np.ones(spec.n_traces)
        spec.extra_events = [
            ((surface_ns + 2.0 * z * math.sqrt(eps) / C) * ones, amp) for z, amp in layers
        ]
        return spec, T_SRC_NS + surface_ns

    scenarios = []
    for h in (0.25, 0.3, 0.4, 0.5):
        scenarios.append((f"DS_h{h:.2f}", "D+S", h, 3.0, -2.0, []))
    scenarios.append(("DS_h0.30_surface_strong", "D+S", 0.3, 1.0, -2.0, []))
    for z in (0.10, 0.15, 0.20, 0.30):
        for amp in (-0.6, -1.6, 0.6):
            scenarios.append((f"BL_h0.05_z{z:.2f}_a{amp:+.1f}", "B+L", 0.05, 3.0, -2.0, [(z, amp)]))
    # Layer gap 2 z sqrt(eps) / c = 1.6-2.4 ns after the surface keeps it inside
    # the window (strongest event + 2 h_max / c + 0.5 / f_c); an opposite-
    # polarity layer splits from 0.75 / f_c, a same-polarity one only further.
    for z, amp in ((0.08, 0.6), (0.08, 1.6), (0.11, -1.6), (0.12, -0.6)):
        scenarios.append((f"DSL_h0.30_z{z:.2f}_a{amp:+.1f}", "D+S+L", 0.3, 3.0, -2.0, [(z, amp)]))

    rows = []
    for name, kind, h, direct, surface, layers in scenarios:
        spec, truth_ns = spec_for(h, direct, surface, layers)
        radargram = make_radargram(spec, line_id=name)
        filtered = prepare_fitting_branch(radargram).filtered
        for variant in VARIANTS:
            for h_known in ((h, None) if variant == "first_h" else (None,)):
                with use_variant(variant, h_of=lambda r, hk=h_known: hk):
                    zero = tz.find_time_zero(radargram, filtered)
                dz = zero.t_zero_ns - truth_ns
                rows.append(
                    {
                        "scenario": name,
                        "kind": kind,
                        "h_m": h,
                        "layers": ";".join(f"z{z}:a{a}" for z, a in layers),
                        "variant": variant + ("" if variant != "first_h" else ("_known" if h_known else "_unknown")),
                        "n_events": len(zero.events_ns),
                        "events_ns": ";".join(f"{t:.3f}" for t in zero.events_ns),
                        "zero_rule": zero.zero_rule,
                        "t_zero_ns": round(zero.t_zero_ns, 4),
                        "truth_zero_ns": round(truth_ns, 4),
                        "dt_zero_ns": round(dz, 4),
                        "depth_err_cm": round(-100.0 * v * dz / 2.0, 2),
                        "within_2cm": abs(v * dz / 2.0) <= 0.02,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "m29e2_two_event_zero_analytic.csv", index=False)
    multi = table[table.n_events >= 2]
    summary = {
        "scenarios": len(scenarios),
        "multi_event_scenarios": int(multi.scenario.nunique()),
        "pass_2cm_multi_event": multi.groupby("variant")["within_2cm"].agg(["sum", "size"]).to_dict("index"),
        "pass_2cm_by_kind": multi.groupby(["kind", "variant"])["within_2cm"].agg(["sum", "size"]).reset_index().to_dict("records"),
        "eps_r": eps, "f_c_hz": f_c, "offset_m": s, "seed": 0,
    }
    return summary


# --------------------------------------------------------------- gprMax part


def gprmax_part() -> dict[str, object]:
    import eval_depth_synthetic as es
    from gpr_detection.io.synthetic import load_synthetic_truth

    heights = {}

    def h_of(radargram):
        model_id = radargram.metadata.get("model_id", radargram.line_id)
        if model_id not in heights:
            heights[model_id] = load_synthetic_truth(model_id).air_gap_m
        h = heights[model_id]
        return h if h > 0 else None

    rows, summary = [], {}
    for variant in VARIANTS:
        STATE["log"] = []
        started = time.perf_counter()
        with use_variant(variant, h_of=h_of):
            runs = es.evaluate_all()
            checks = es.regression_checks(runs)
        failed = [f"{c.check_id}/{c.model_id}: {c.measured}" for c in checks if not c.passed]
        summary[variant] = {
            "n_checks": len(checks),
            "n_pass": sum(c.passed for c in checks),
            "failed": failed,
            "multi_event_choices": list(STATE["log"]),
            "seconds": round(time.perf_counter() - started, 1),
        }
        for run in runs:
            row = es.model_row(run)
            zero = run.inputs.zero
            rows.append(
                {
                    "variant": variant,
                    "model_id": row["model_id"],
                    "air_gap_m": run.truth.air_gap_m,
                    "zero_rule": zero.zero_rule,
                    "n_events": len(zero.events_ns),
                    "events_ns": ";".join(f"{t:.3f}" for t in zero.events_ns),
                    "t_zero_ns": zero.t_zero_ns,
                    "h_est_m": zero.h_est_m,
                    "d_est_m": row["d_est_m"],
                    "d_err_cm": row["d_err_cm"],
                    "v_err_pct": row["v_err_pct"],
                    "eps_qc": row["eps_qc"],
                    "status": row["status"],
                }
            )
    table = pd.DataFrame(rows).replace("", np.nan)
    base = table[table.variant == "current"].set_index("model_id")
    for col in ("t_zero_ns", "d_est_m"):
        table[f"delta_{col}"] = table[col].astype(float).to_numpy() - base.loc[
            table["model_id"], col
        ].astype(float).to_numpy()
    table.to_csv(
        DOCS / "m29e2_two_event_zero_gprmax.csv", index=False, float_format="%.5f"
    )
    for variant in VARIANTS:
        sub = table[table.variant == variant]
        summary[variant]["max_abs_delta_depth_mm"] = float(np.nanmax(np.abs(sub["delta_d_est_m"]))) * 1000
        summary[variant]["models_with_zero_change"] = sub.loc[
            sub["delta_t_zero_ns"].abs() > 1e-9, "model_id"
        ].tolist()
    # M20 event amplitudes (2-D synthetic: reported, not a physics criterion).
    from gpr_detection.io.synthetic import load_synthetic

    radargram = load_synthetic("M20")
    events = tz.detect_zero_events(
        prepare_fitting_branch(radargram, DEFAULT_DEPTH_CONFIG.bg_window).filtered,
        radargram.dt_ns, radargram.f_c_hz, antenna_offset_m=radargram.antenna_offset_m,
        config=DEFAULT_DEPTH_CONFIG.zero_config(),
    )
    summary["M20_events"] = {
        "events_ns": [round(t, 4) for t in events.events_ns],
        "rel_amplitude": [round(a / max(events.amplitudes), 4) for a in events.amplitudes],
        "gap_ns": round(events.events_ns[-1] - events.events_ns[0], 4),
        "gap_h_ns_json": round(gap_for_height_ns(0.3, radargram.antenna_offset_m), 4),
        "h_est_m": events.h_est_m,
    }
    return summary


# --------------------------------------------------------------- 4TU part


def _track(env: np.ndarray, t_ns: float, dt_ns: float) -> tuple[float, float, bool]:
    """Local envelope maximum within +-TRACK_HALF_NS of ``t_ns``: (time, amp, interior)."""
    half = int(round(TRACK_HALF_NS / dt_ns))
    centre = int(round(t_ns / dt_ns))
    lo, hi = max(centre - half, 1), min(centre + half, len(env) - 2)
    if hi <= lo:
        return math.nan, math.nan, False
    k = lo + int(np.argmax(env[lo : hi + 1]))
    interior = lo < k < hi
    kk, amp = parabolic_peak(env, k)
    return kk * dt_ns, amp, interior


def _signed_polarity(trace: np.ndarray, t_ns: float, dt_ns: float, period_ns: float) -> int:
    half = max(1, int(round(0.25 * period_ns / dt_ns)))
    centre = int(round(t_ns / dt_ns))
    lo, hi = max(centre - half, 0), min(centre + half, len(trace) - 1)
    seg = trace[lo : hi + 1]
    return int(np.sign(seg[int(np.argmax(np.abs(seg)))]))


def fourtu_part() -> dict[str, object]:
    from gpr_detection.io.fourtu import T0Choice, load_fourtu_radargram
    from gpr_detection.io.paths import load_dataset_paths

    root = load_dataset_paths().fourtu
    survey = pd.read_csv(SURVEY_CSV, dtype={"activity_id": str, "path": str})
    late = survey[survey.zero_rule == "air_surface_late"]
    activities = sorted(set(late.activity_id))
    lines = survey[survey.activity_id.isin(activities)]
    single_ref = (
        survey[survey.zero_rule == "air_single_event"]
        .assign(t=lambda d: d.events_ns.astype(float))
        .groupby("activity_id")["t"]
        .agg(["median", "min", "max", "size"])
    )
    cfg = DEFAULT_DEPTH_CONFIG
    rows = []
    for rec in lines.itertuples(index=False):
        radargram = load_fourtu_radargram(root, rec.activity_id, rec.path, T0Choice.BOTH)
        filtered = prepare_fitting_branch(radargram, cfg.bg_window).filtered
        events = tz.detect_zero_events(
            filtered, radargram.dt_ns, radargram.f_c_hz, pick_mode=cfg.pick_mode,
            antenna_offset_m=radargram.antenna_offset_m, config=cfg.zero_config(),
        )
        ev = list(events.events_ns)
        csv_ev = [float(t) for t in str(rec.events_ns).split(";")]
        if not np.allclose(ev, csv_ev, atol=6e-4):
            raise RuntimeError(f"{rec.activity_id}/{rec.path}: events differ from survey csv")
        dt = radargram.dt_ns
        period = period_ns_or_default(radargram.f_c_hz, tz.DEFAULT_PERIOD_NS)
        v_meta = C / math.sqrt(radargram.eps_r_meta)
        row = {
            "activity_id": rec.activity_id, "site": rec.site, "path": rec.path,
            "activity_has_gt": rec.activity_has_gt, "n_candidates": rec.n_candidates,
            "dt_ns": dt, "eps_r_used": radargram.eps_r_meta, "n_traces": filtered.shape[1],
            "n_events": len(ev), "events_ns": ";".join(f"{t:.3f}" for t in ev),
            "rel_amp": ";".join(f"{a / max(events.amplitudes):.3f}" for a in events.amplitudes),
        }
        if len(ev) >= 2:
            t_by = {}
            for variant in VARIANTS:
                index, reason = choose_index(variant, events, radargram, None)
                t_by[variant] = ev[index]
                row[f"idx_{variant}"] = index
            for variant in ("first", "strongest", "first_h"):
                row[f"dd_{variant}_m"] = v_meta * (t_by["current"] - t_by[variant]) / 2.0
            ref = single_ref.loc[rec.activity_id] if rec.activity_id in single_ref.index else None
            if ref is not None:
                row |= {
                    "single_ref_median_ns": ref["median"], "single_ref_min_ns": ref["min"],
                    "single_ref_max_ns": ref["max"], "single_ref_n": int(ref["size"]),
                    "first_minus_ref_ns": ev[0] - ref["median"],
                    "last_minus_ref_ns": ev[-1] - ref["median"],
                    "strongest_minus_ref_ns": ev[int(np.argmax(events.amplitudes))] - ref["median"],
                }
            # relative polarity of the mean trace at each event
            mean_trace = filtered.mean(axis=1)
            pol = [_signed_polarity(mean_trace, t, dt, period) for t in ev]
            row["polarity"] = ";".join(f"{p:+d}" for p in pol)
            row["last_same_polarity_as_first"] = pol[-1] == pol[0]
        # per-group kinematics (11-trace stacks)
        n_groups = filtered.shape[1] // GROUP_TRACES
        t_groups = {i: [] for i in range(len(ev))}
        present = {i: [] for i in range(len(ev))}
        for g in range(n_groups):
            stack = filtered[:, g * GROUP_TRACES : (g + 1) * GROUP_TRACES].mean(axis=1)
            env = pick_signal(stack, "envelope")
            window_max = float(env[: events.window_end_k + 1].max())
            for i, t in enumerate(ev):
                tt, amp, interior = _track(env, t, dt)
                t_groups[i].append(tt)
                present[i].append(bool(interior and amp >= cfg.zero_config().height_rel * window_max))
        row["n_groups"] = n_groups
        if n_groups >= 3:
            t0 = np.asarray(t_groups[0])
            row["std_first_ns"] = float(np.nanstd(t0))
            row["present_first"] = float(np.mean(present[0]))
            if len(ev) >= 2:
                tl = np.asarray(t_groups[len(ev) - 1])
                ok = np.isfinite(t0) & np.isfinite(tl)
                row["std_last_ns"] = float(np.nanstd(tl))
                row["std_gap_ns"] = float(np.nanstd(tl - t0))
                row["present_last"] = float(np.mean(present[len(ev) - 1]))
                if ok.sum() >= 3 and np.std(t0[ok]) > 0 and np.std(tl[ok]) > 0:
                    row["corr_last_first"] = float(np.corrcoef(t0[ok], tl[ok])[0, 1])
                    row["slope_last_on_first"] = float(np.polyfit(t0[ok], tl[ok], 1)[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "m29e2_two_event_zero_fourtu.csv", index=False, float_format="%.4f")

    two = table[table.n_events >= 2]
    single = table[table.n_events == 1]

    def q(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return np.percentile(s, [0, 25, 50, 75, 100]).round(3).tolist() if len(s) else []

    gt = two[two.activity_has_gt]
    summary = {
        "activities": activities,
        "lines": len(table), "two_event_lines": len(two), "single_lines_same_activities": len(single),
        "two_event_gt_lines": len(gt),
        "first_is_strongest": int((two.idx_strongest == 0).sum()),
        "first_is_strongest_gt": int((gt.idx_strongest == 0).sum()),
        "variant_change_vs_current_lines": {
            v: int((two[f"idx_{v}"] != two["idx_current"]).sum()) for v in ("first", "strongest", "first_h")
        },
        "dd_first_m_q": q(two["dd_first_m"]), "dd_first_m_q_gt": q(gt["dd_first_m"]),
        "dd_strongest_m_q": q(two["dd_strongest_m"]),
        "first_vs_strongest_differ_lines": int((two.idx_first != two.idx_strongest).sum()),
        "first_vs_strongest_differ_gt_lines": int((gt.idx_first != gt.idx_strongest).sum()),
        "abs_first_minus_ref_q": q(two["first_minus_ref_ns"].abs()),
        "abs_last_minus_ref_q": q(two["last_minus_ref_ns"].abs()),
        "abs_strongest_minus_ref_q": q(two["strongest_minus_ref_ns"].abs()),
        "first_closer_to_ref_than_last": int(
            (two["first_minus_ref_ns"].abs() < two["last_minus_ref_ns"].abs()).sum()
        ),
        "lines_with_ref": int(two["first_minus_ref_ns"].notna().sum()),
        "gt_first_closer": int((gt["first_minus_ref_ns"].abs() < gt["last_minus_ref_ns"].abs()).sum()),
        "last_same_polarity_as_first": int(two["last_same_polarity_as_first"].sum()),
        "std_first_ns_q_two": q(two.get("std_first_ns")), "std_first_ns_q_single": q(single.get("std_first_ns")),
        "std_last_ns_q": q(two.get("std_last_ns")), "std_gap_ns_q": q(two.get("std_gap_ns")),
        "present_last_q": q(two.get("present_last")),
        "slope_last_on_first_q": q(two.get("slope_last_on_first")),
        "corr_last_first_q": q(two.get("corr_last_first")),
        "group_traces": GROUP_TRACES, "track_half_ns": TRACK_HALF_NS,
        "gt_use": "activity membership copied from m29e_zero_survey_lines.csv (no depth read)",
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("analytic", "gprmax", "fourtu"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    summary = {"analytic": analytic_part, "gprmax": gprmax_part, "fourtu": fourtu_part}[args.part]()
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    (DOCS / f"m29e2_two_event_zero_{args.part}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
