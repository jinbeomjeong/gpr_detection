"""E3x plan-defect diagnostics (gpr-processing-expert, read-only, no detector).

Two questions from the E3x plan stage (``runs/eval/m33/e3x/plan``):

1. shift_mismatch: is the shifted band an exact translation of the native
   one, and where does the 1-sample step between ``row0 + n`` and the
   re-measured ``row0'`` come from (record-top effects on the native field,
   steps (1)-(3), the envelope)?  For every band model on the mismatch lines
   and every target row, record ``step = row0' - (row0 + n)``.
2. r_edge: compare the implemented definition (max over all rows of the
   aperture-taper columns of the untapered field) with variants: the on-off
   difference field ``untapered (1 - w)``, the outermost column only, the
   first-reflection rows only, and the synthetic grid before the H filter.

Writes json/csv to ``analysis/processing/m33_e3x_plandefect_diag*.{json,csv}``.
Only counts and amplitudes of synthetic patches; no detector output is read.
"""

import csv
import json
import math
import sys
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
from gpr_detection.preprocess.filters import bandpass, dewow, remove_dc  # noqa: E402

OUT = ROOT / "analysis" / "processing"
POOL = ROOT / "analysis" / "detection" / "m33_e3x_pool.csv"

MISMATCH = [
    ("010.22/Path1", "M01", 28.0),
    ("010.22/Path4", "M08", 47.0),
    ("02.3/Path2", "M01", 68.0),
    ("07.3/Path1", "M01", 68.0),
    ("pipe/1030_190118AC/LID10001", "M01", 28.0),
    ("pipe/1030_190118AC/LID10001", "M08", 28.0),
    ("pipe/1030_190313AA/LID10004", "M09", 28.0),
]
R_EDGE_LINES = ["011.13/Profile_6149", "01.7/Path3", "010.17/Path1", "pipe/1030_190118AC/LID10001"]


def pool_lines():
    return {line.line_id: line for line in e3x.read_pool_lines(POOL)}


def load(line):
    data = runner.load_line(line.ref)
    return data, inject.prepare_line_context(data.radargram)


def run_of(column_env, peak_row):
    """Top row of the -20 dB run seeded at peak_row (1-D)."""
    thr = inject.WING_AMPLITUDE * column_env[peak_row]
    r = peak_row
    while r - 1 >= 0 and column_env[r - 1] >= thr:
        r -= 1
    return r, thr


def stepwise(col, bg):
    dt = bg.dt_ns
    a = remove_dc(col[:, None])
    b = dewow(a, dt, bg.f_c_hz)
    c = bandpass(b, dt, bg.f_c_hz) if bg.f_c_hz is not None else b
    return {"raw": col[:, None], "dc": a, "dewow": b, "bandpass": c}


def shift_diag(lines):
    rows = []
    detail = []
    band_lib = responses_io.load_library()
    wanted = sorted({lid for lid, _, _ in MISMATCH})
    for lid in wanted:
        line = lines[lid]
        data, ctx = load(line)
        responses = e3x.line_responses(band_lib, data, e3x.BAND_MODELS)
        lengths = {a: place.band_length_traces(ctx, p) for a, p in place.ARM_PERIODS.items()}
        for model, resp in responses.items():
            native = inject.build_response_patch(resp, ctx, inject.K_MAIN)
            if isinstance(native, inject.RejectReason):
                continue
            taper = place.position_taper_traces(resp, ctx)
            nb = inject.build_band_patch(native, ctx, lengths["long"], taper)
            if isinstance(nb, inject.RejectReason):
                continue
            row0 = nb.row0
            apex_col = round(resp.apex_trace)
            for row_px in place.TARGET_ROWS_PX:
                n = place.shift_samples_for(ctx, row_px, row0)
                if not place.within_shift_limit(n, ctx.period_samples):
                    continue
                rs = place.row_shift_for(ctx, resp, inject.K_MAIN, row_px, row0)
                if not isinstance(rs, place.RowShift):
                    continue
                sb = inject.build_band_patch(rs.patch, ctx, lengths["long"], taper)
                if isinstance(sb, inject.RejectReason):
                    continue
                tgt = place.target_sample(ctx, row_px)
                step = sb.row0 - (row0 + n)
                # translation exactness of the raw field column and envelope
                f0 = native.field[:, apex_col]
                f1 = rs.patch.field[:, apex_col]
                e0 = native.envelope[:, apex_col]
                e1 = rs.patch.envelope[:, apex_col]
                m = min(len(f0), len(f1) - n) if n >= 0 else None
                rec = {
                    "line_id": lid, "model": model, "row_px": row_px, "n": n,
                    "native_row0": row0, "shifted_row0": sb.row0, "step": step,
                    "target": tgt, "gap": sb.row0 - tgt, "residual_round": row0 + n - tgt,
                    "n_rows_native": len(f0), "n_rows_shifted": len(f1),
                    "native_row0_P": row0 / ctx.period_samples,
                    "mismatch": abs(sb.row0 - tgt) > place.SHIFT_MATCH_SAMPLES,
                }
                if m is not None and m > 0:
                    scale = float(np.abs(f0).max())
                    rec["field_maxdiff_rel"] = float(np.abs(f1[n:n + m] - f0[:m]).max() / scale)
                    escale = float(e0.max())
                    rec["env_maxdiff_rel"] = float(np.abs(e1[n:n + m] - e0[:m]).max() / escale)
                    # threshold margins at native run top
                    pk0 = inject._apex_peak(native.envelope, apex_col, native.apex_sample, ctx.period_samples)
                    pk1 = inject._apex_peak(rs.patch.envelope, apex_col, rs.patch.apex_sample, ctx.period_samples)
                    r0, thr0 = run_of(e0, pk0.seed[0])
                    r1, thr1 = run_of(e1, pk1.seed[0])
                    rec["ratio_native_at_row0m1"] = float(e0[r0 - 1] / pk0.amplitude) if r0 > 0 else None
                    rec["ratio_native_at_row0"] = float(e0[r0] / pk0.amplitude)
                    rec["ratio_shift_at_row0p_m1"] = float(e1[r1 - 1] / pk1.amplitude)
                    rec["ratio_shift_at_row0p"] = float(e1[r1] / pk1.amplitude)
                    rec["ratio_shift_at_row0+n"] = float(e1[row0 + n] / pk1.amplitude)
                    rec["ratio_shift_at_row0+n-1"] = float(e1[row0 + n - 1] / pk1.amplitude)
                    rec["peak_rel_diff"] = float(pk1.amplitude / pk0.amplitude - 1.0)
                    if (lid, model, row_px) in {(a, b, c) for a, b, c in MISMATCH}:
                        # which step breaks the translation? use the raw field column
                        s0 = stepwise(f0, ctx.background)
                        s1 = stepwise(f1, ctx.background)
                        per = {}
                        for key in s0:
                            a0 = s0[key][:, 0]
                            a1 = s1[key][:, 0]
                            per[key] = float(np.abs(a1[n:n + m] - a0[:m]).max() / np.abs(a0).max())
                        rec["stepwise_maxdiff_rel"] = per
                        # where in the native column is the energy relative to the top
                        a0 = s0["bandpass"][:, 0]
                        top = np.abs(a0[: max(1, int(ctx.period_samples))]).max() / np.abs(a0).max()
                        rec["native_bp_top1P_rel"] = float(top)
                        rec["raw_row0_rel"] = float(abs(f0[0]) / np.abs(f0).max())
                        rec["shift_ns_native"] = float(inject._alignment_shift_ns(resp, ctx))
                        detail.append(rec)
                rows.append(rec)
    return rows, detail


def first_reflection_rows(patch, ctx):
    resp = patch.response
    tau_m = resp.apex_time_ns - resp.time_origin_ns
    last = patch.apex_sample + (tau_m - ctx.period_ns) / ctx.dt_ns
    return last


def r_edge_diag(lines):
    lib = e3x.load_hyperbola_library()
    out = []
    for lid in R_EDGE_LINES:
        line = lines[lid]
        data, ctx = load(line)
        responses = e3x.line_responses(lib, data, e3x.HYPERBOLA_MODELS)
        for model, resp in responses.items():
            patch = inject.build_response_patch(resp, ctx, inject.K_MAIN)
            if isinstance(patch, inject.RejectReason):
                continue
            bg = ctx.background
            n_t = patch.n_aperture_taper
            w = inject.edge_taper_weights(resp.n_traces, n_t)
            unt = place.untapered_field(patch)
            env_unt = place.envelope(inject.linear_steps(unt, bg))
            diff = unt - patch.field  # off - on = unt (1 - w)
            env_diff = place.envelope(inject.linear_steps(diff, bg))
            apex_col = round(resp.apex_trace)
            apex = place.apex_envelope_peak(env_unt, apex_col, patch.apex_sample, ctx.period_samples)
            cols = np.r_[0:n_t, resp.n_traces - n_t:resp.n_traces]
            impl = env_unt[:, cols].max()
            imax = np.unravel_index(np.argmax(env_unt[:, cols]), env_unt[:, cols].shape)
            outer = env_unt[:, [0, resp.n_traces - 1]].max()
            dmax = env_diff.max()
            dpos = np.unravel_index(np.argmax(env_diff), env_diff.shape)
            last_fr = first_reflection_rows(patch, ctx)
            fr = env_diff[: max(1, math.floor(last_fr)) + 1, :].max()
            impl_fr = env_unt[: max(1, math.floor(last_fr)) + 1][:, cols].max()
            # synthetic grid, 2.5-D weight on, no H filter, envelope only
            syn = inject._time_weighted(resp, True)
            syn_env = place.envelope(syn)
            ap_s = resp.apex_time_ns / resp.dt_ns
            p_s = ctx.period_ns / resp.dt_ns
            syn_apex = place.apex_envelope_peak(syn_env, apex_col, ap_s, p_s)
            syn_outer = syn_env[:, [0, resp.n_traces - 1]].max()
            syn_cols = syn_env[:, cols].max()
            db = lambda x, ref: 20 * math.log10(x / ref) if x > 0 else float("-inf")
            out.append({
                "line_id": lid, "model": model, "dt_ns": bg.dt_ns, "period_ns": ctx.period_ns,
                "n_aperture_taper": n_t, "apex_depth_P": (patch.apex_sample - ctx.zero.k_zero) / ctx.period_samples if hasattr(ctx.zero, "k_zero") else None,
                "r_edge_impl_db": db(impl, apex),
                "r_edge_impl_check_db": place.r_edge_db(patch, ctx),
                "impl_argmax_col_in_taper": int(imax[1]) if imax[1] < n_t else int(imax[1] - n_t),
                "impl_argmax_side": "left" if imax[1] < n_t else "right",
                "impl_argmax_row_minus_apex_P": (imax[0] - patch.apex_sample) / ctx.period_samples,
                "r_edge_outer_col_db": db(outer, apex),
                "r_edge_diff_db": db(dmax, apex),
                "diff_argmax_row_minus_apex_P": (dpos[0] - patch.apex_sample) / ctx.period_samples,
                "diff_argmax_col": int(dpos[1]),
                "r_edge_diff_first_reflection_db": db(fr, apex),
                "r_edge_impl_first_reflection_db": db(impl_fr, apex),
                "syn_outer_db": db(syn_outer, syn_apex),
                "syn_taper_cols_db": db(syn_cols, syn_apex),
                "half_width_20db": float(patch.half_width_20db),
                "half_traces": (resp.n_traces - 1) / 2,
            })
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], dict):
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    lines = pool_lines()
    rows, detail = shift_diag(lines)
    write_csv(OUT / "m33_e3x_plandefect_diag_shift.csv", rows)
    redge = r_edge_diag(lines)
    write_csv(OUT / "m33_e3x_plandefect_diag_redge.csv", redge)
    steps = {}
    for r in rows:
        steps[str(r["step"])] = steps.get(str(r["step"]), 0) + 1
    summary = {
        "n_shift_tests": len(rows),
        "step_counts": steps,
        "n_mismatch_1sample": sum(1 for r in rows if r["mismatch"]),
        "n_gap_over_1p5": sum(1 for r in rows if abs(r["gap"]) > 1.5),
        "max_abs_gap": max(abs(r["gap"]) for r in rows),
        "field_maxdiff_rel_max": max((r.get("field_maxdiff_rel", 0.0) for r in rows)),
        "env_maxdiff_rel_max": max((r.get("env_maxdiff_rel", 0.0) for r in rows)),
        "mismatch_detail": detail,
    }
    (OUT / "m33_e3x_plandefect_diag.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "n_shift_tests": len(rows), "n_redge": len(redge)}))


if __name__ == "__main__":
    main()
