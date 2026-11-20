"""E3x plan-defect diagnostics, part 2 (gpr-processing-expert, no detector).

(a) r_edge of every planned hyperbola (line, model) under the implemented
    definition (max of the untapered envelope over the aperture-taper
    columns) and under the on-off difference field ``untapered (1 - w)``
    (all rows), with the strata each gives (plan §2-3 thresholds) and the
    position / cluster / Kish counts of the manipulated and negative-control
    sets.
(b) Band arms of every planned band position: is each arm a pure translation
    of the others (record-top truncation of the synthetic window, steps
    (1)-(3) edge effects)?  Per arm: line row of synthetic time 0, raw apex
    column at row 0, envelope difference to the least-truncated arm of the
    same position (aligned), and the top-row step.

Reads only plan tables (no detector output) and the background lines.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "detection"))

import m33_e3x as e3x  # noqa: E402
import m33_e3x_place as place  # noqa: E402
import m32_pipe_responses as responses_io  # noqa: E402
import m32_recipe_eval as runner  # noqa: E402

from gpr_detection.eval import inject  # noqa: E402

OUT = ROOT / "analysis" / "processing"
PLAN = ROOT / "runs" / "eval" / "m33" / "e3x" / "plan"
POOL = ROOT / "analysis" / "detection" / "m33_e3x_pool.csv"


def read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def stratum(r):
    s = tuple(x for x in (12.0, 18.0) if r + x >= 0.0)
    if s:
        return "manipulated:" + ",".join(f"{x:g}" for x in s)
    if r + 18.0 < -6.0:
        return "negative_control"
    return "gray"


def kish(counts):
    tot = sum(counts)
    return tot * tot / sum(c * c for c in counts) if counts else None


def cluster_stats(rows):
    out = {}
    for ds in ("4tu", "gz_ids"):
        sub = [r for r in rows if r["dataset"] == ds]
        c = Counter(r["cluster_id"] for r in sub)
        out[ds] = {"positions": len(sub), "clusters": len(c), "g_eff": kish(list(c.values()))}
    out["positions_total"] = len(rows)
    out["clusters_total"] = len({(r["dataset"], r["cluster_id"]) for r in rows})
    return out


def r_edge_pair(patch, ctx):
    bg = ctx.background
    unt = place.untapered_field(patch)
    env_unt = place.envelope(inject.linear_steps(unt, bg))
    env_diff = place.envelope(inject.linear_steps(unt - patch.field, bg))
    resp = patch.response
    n_t = patch.n_aperture_taper
    apex = place.apex_envelope_peak(env_unt, round(resp.apex_trace), patch.apex_sample, ctx.period_samples)
    cols = np.r_[0:n_t, resp.n_traces - n_t:resp.n_traces]
    impl = float(env_unt[:, cols].max())
    diff = float(env_diff.max())
    outer = float(env_unt[:, [0, resp.n_traces - 1]].max())
    db = lambda x: 20 * math.log10(x / apex)
    return db(impl), db(diff), db(outer)


def part_a(pool):
    hyp = read(PLAN / "hyperbola_positions.csv")
    lib = e3x.load_hyperbola_library()
    by_line = defaultdict(set)
    for r in hyp:
        by_line[r["line_id"]].add(r["model"])
    values = {}
    for lid, models in sorted(by_line.items()):
        data = runner.load_line(pool[lid].ref)
        ctx = inject.prepare_line_context(data.radargram)
        responses = e3x.line_responses(lib, data, e3x.HYPERBOLA_MODELS)
        for model in sorted(models):
            patch = inject.build_response_patch(responses[model], ctx, inject.K_MAIN)
            values[(lid, model)] = r_edge_pair(patch, ctx)
    rows = []
    for r in hyp:
        impl, diff, outer = values[(r["line_id"], r["model"])]
        rows.append({
            "position_id": r["position_id"], "line_id": r["line_id"], "dataset": r["dataset"],
            "cluster_id": r["cluster_id"], "model": r["model"],
            "r_edge_plan_db": float(r["r_edge_db"]), "r_edge_impl_db": impl,
            "r_edge_diff_db": diff, "r_edge_outer_db": outer,
            "stratum_plan": r["edge_stratum"] + (":" + r["edge_snr_set"] if r["edge_snr_set"] else ""),
            "stratum_impl": stratum(impl), "stratum_diff": stratum(diff),
            "boundary_unit_diff": any(abs(diff + s) < 1.5 for s in (12.0, 18.0)),
        })
    comp = defaultdict(Counter)
    for r in rows:
        comp[r["model"]][r["stratum_diff"]] += 1
    man = [r for r in rows if r["stratum_diff"].startswith("manipulated")]
    neg = [r for r in rows if r["stratum_diff"] == "negative_control"]
    only18 = [r for r in man if r["stratum_diff"] == "manipulated:18"]
    by_model_diff = defaultdict(list)
    for r in rows:
        by_model_diff[r["model"]].append(r["r_edge_diff_db"])
    summary = {
        "n_positions": len(rows),
        "plan_matches_impl_recompute": all(abs(r["r_edge_plan_db"] - r["r_edge_impl_db"]) < 1e-9 for r in rows),
        "max_abs_diff_minus_outer_db": max(abs(r["r_edge_diff_db"] - r["r_edge_outer_db"]) for r in rows),
        "strata_diff": dict(Counter(r["stratum_diff"].split(":")[0] for r in rows)),
        "strata_diff_detail": dict(Counter(r["stratum_diff"] for r in rows)),
        "composition_by_model_diff": {m: dict(c) for m, c in sorted(comp.items())},
        "r_edge_diff_by_model": {m: {"median": float(np.median(v)), "min": min(v), "max": max(v)} for m, v in sorted(by_model_diff.items())},
        "manipulated": cluster_stats(man),
        "manipulated_models": sorted({r["model"] for r in man}),
        "manipulated_only18_share": len(only18) / len(man) if man else None,
        "negative_control": cluster_stats(neg),
        "negative_control_models": sorted({r["model"] for r in neg}),
        "boundary_units_diff": sum(1 for r in rows if r["boundary_unit_diff"]),
        "shift_impl_minus_diff_db": {"median": float(np.median([r["r_edge_impl_db"] - r["r_edge_diff_db"] for r in rows])),
                                     "min": min(r["r_edge_impl_db"] - r["r_edge_diff_db"] for r in rows),
                                     "max": max(r["r_edge_impl_db"] - r["r_edge_diff_db"] for r in rows)},
    }
    return rows, summary


def part_b(pool):
    pos = read(PLAN / "band_positions.csv")
    arms = read(PLAN / "band_arms.csv")
    lib = responses_io.load_library()
    arms_of = defaultdict(list)
    for a in arms:
        arms_of[a["position_id"]].append(a)
    out = []
    by_line = defaultdict(list)
    for p in pos:
        by_line[p["line_id"]].append(p)
    for lid, plist in sorted(by_line.items()):
        data = runner.load_line(pool[lid].ref)
        ctx = inject.prepare_line_context(data.radargram)
        responses = e3x.line_responses(lib, data, e3x.BAND_MODELS)
        for p in plist:
            resp = responses[p["model"]]
            taper = int(p["taper_traces"])
            built = {}
            for a in arms_of[p["position_id"]]:
                n = int(a["shift_samples"])
                if n in built:
                    continue
                moved = place.shifted_response(resp, n * ctx.dt_ns)
                patch = inject.build_response_patch(moved, ctx, inject.K_MAIN)
                band = inject.build_band_patch(patch, ctx, int(a["length_traces"]), taper)
                col = band.field[:, 0 if False else band.n_columns // 2]
                env = place.envelope(inject.linear_steps(col[:, None], ctx.background))[:, 0]
                start_row = inject._alignment_shift_ns(moved, ctx) / ctx.dt_ns
                built[n] = (a["bin_target_px"], band, col, env, start_row)
            ref_n = max(built, key=lambda k: built[k][4])
            _, ref_band, ref_col, ref_env, ref_start = built[ref_n]
            pk = ref_env.max()
            for n, (row_px, band, col, env, start_row) in sorted(built.items(), key=lambda kv: kv[0]):
                d = n - ref_n
                # align env rows r (arm) with r - d (ref)
                lo = max(0, d)
                hi = min(len(env), len(ref_env) + d)
                diff = np.abs(env[lo:hi] - ref_env[lo - d:hi - d])
                r0, r1 = band.row0, band.row1
                w0, w1 = max(lo, r0 - int(ctx.period_samples)), min(hi, r1 + int(ctx.period_samples))
                truth_diff = float(np.abs(env[w0:w1] - ref_env[w0 - d:w1 - d]).max() / pk) if w1 > w0 else None
                out.append({
                    "position_id": p["position_id"], "line_id": lid, "dataset": p["line_id"].startswith("pipe/") and "gz_ids" or "4tu",
                    "model": p["model"], "row_px": row_px, "shift_samples": n,
                    "synthetic_t0_line_row": start_row,
                    "truncated_top": start_row < 0,
                    "raw_row0_rel": float(abs(col[0]) / np.abs(col).max()),
                    "env_maxdiff_rel_to_ref": float(diff.max() / pk),
                    "env_truthwin_maxdiff_rel_to_ref": truth_diff,
                    "top_step_vs_ref": band.row0 - (ref_band.row0 + d),
                    "ref_shift": ref_n,
                })
    trunc = [r for r in out if r["truncated_top"]]
    summary = {
        "n_arm_builds": len(out),
        "n_truncated_top": len(trunc),
        "n_truncated_raw_row0_over_1e-3": sum(1 for r in trunc if r["raw_row0_rel"] > 1e-3),
        "truncated_rows": Counter(r["row_px"] for r in trunc),
        "max_env_truthwin_diff": max((r["env_truthwin_maxdiff_rel_to_ref"] or 0.0) for r in out),
        "n_truthwin_diff_over_0.01": sum(1 for r in out if (r["env_truthwin_maxdiff_rel_to_ref"] or 0) > 0.01),
        "n_truthwin_diff_over_0.05": sum(1 for r in out if (r["env_truthwin_maxdiff_rel_to_ref"] or 0) > 0.05),
        "top_step_counts": Counter(str(r["top_step_vs_ref"]) for r in out),
        "worst": sorted(out, key=lambda r: -(r["env_truthwin_maxdiff_rel_to_ref"] or 0))[:12],
    }
    summary["truncated_rows"] = dict(summary["truncated_rows"])
    summary["top_step_counts"] = dict(summary["top_step_counts"])
    return out, summary


def write_csv(path, rows):
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    pool = {line.line_id: line for line in e3x.read_pool_lines(POOL)}
    a_rows, a_sum = part_a(pool)
    write_csv(OUT / "m33_e3x_plandefect_diag2_redge.csv", a_rows)
    b_rows, b_sum = part_b(pool)
    write_csv(OUT / "m33_e3x_plandefect_diag2_band.csv", b_rows)
    (OUT / "m33_e3x_plandefect_diag2.json").write_text(
        json.dumps({"r_edge": a_sum, "band": b_sum}, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "n_hyp": len(a_rows), "n_band_arms": len(b_rows)}))


if __name__ == "__main__":
    main()
