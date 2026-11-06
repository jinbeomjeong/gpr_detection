"""M2-9 C step 1b: curvature-v QC of the reviewed apex candidates (after classification).

Run only after ``data/processing/m29c_gallery/review_sheet_locked.csv`` is fixed.
Ground-truth depth tables and M2-8 matching results are NOT read.

Per candidate (same path as scripts/eval_depth_4tu.py, DepthConfig defaults):
  box = eval_depth_4tu.candidate_box, picks = extract_picks(apex_hint = candidate),
  curvature = fit_curvature(picks, R = 0, air_gap = "ground")  -- QC ONLY: the
  straight-ray fit is refused for air-coupled data in the depth path; here the
  ground label is passed on purpose and the result never becomes a depth.
  r = v_curv / v_meta,  v_meta = c / sqrt(eps_r_used).
Rule under test (2026-10-31): |r - 1| <= 0.30 -> in, else out.
``curvature_v_unknown`` reasons (fixed before the run):
  no_box (tau missing), no_apex (no apex candidate / hint unmatched / zero zone),
  insufficient_picks (< 9 picks or < 3 per side), fit_failed (not converged or on
  a bound), unstable (sigma_v / v_meta > 0.15: a 2-sigma interval wider than the
  whole +-30 % band, or inlier_frac < 0.5).
Extra shape metrics for alternatives (not used by the rule): RMS of the free-v
fit, RMS with v fixed to v_meta (x0, t0 free), one-sided v (left / right picks,
x0 fixed), pick aperture, and cross-line support (other lines of the activity
with any candidate whose metadata depth is within 0.10 m).

Also writes the refraction-bias table: straight-ray R = 0 fit of exact
air-gap ray times (refraction.reflection_times), zero = surface reflection.

Outputs: docs/processing/m29c_apex_review.csv, m29c_refraction_bias.csv,
m29c_summary.json, m29c_r_by_type.png, m29c_types_montage.png.

Run: conda run -n gpr-dev python -I analysis/processing/m29c_curvature.py
"""

import json
from types import SimpleNamespace
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

from eval_depth_4tu import candidate_box  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import (  # noqa: E402
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.hyperbola import (  # noqa: E402
    fit_curvature,
    fit_hyperbola,
    hyperbola_time,
    ransac_init,
)
from gpr_detection.depth.picks import ApexHint, extract_picks  # noqa: E402
from gpr_detection.depth.refraction import (  # noqa: E402
    reflection_times,
    surface_reflection_time,
)
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.filters import agc  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
GALLERY = PROJECT / "data" / "processing" / "m29c_gallery"
DOCS = PROJECT / "docs" / "processing"
CANDIDATES = DOCS / "fourtu_apex_candidates_v1.csv"

R_TOL = 0.30
UNSTABLE_SIGMA_REL = 0.15
UNSTABLE_INLIER = 0.5
SUPPORT_DEPTH_M = 0.10
MIN_SIDE_FIT = 4
TRUE_TYPES = ("T",)
TYPE_ORDER = ["T", "W", "L", "B", "C", "X", "A", "N", "U"]
TYPE_NAMES = {
    "T": "true\napex",
    "W": "apex\noffset",
    "L": "layer",
    "B": "finite\nband",
    "C": "short arc\nclutter",
    "X": "limb\ncrossing",
    "A": "data\nseam",
    "N": "record\nend",
    "U": "unde-\ncidable",
}


def side_velocity(x, tau, x0, t0, v0):
    """Free (t0, v) fit with x0 fixed, soft-L1 like the main fit."""
    if len(x) < MIN_SIDE_FIT:
        return math.nan
    t_hi = float(tau.max()) + 1.0
    start = [min(max(t0, 1e-3), t_hi - 1e-3), min(max(v0, 0.031), 0.29)]
    sol = least_squares(
        lambda p: hyperbola_time(x, x0, p[0], p[1], 0.0) - tau,
        start,
        bounds=([0.0, 0.03], [t_hi, 0.2998]),
        loss="soft_l1",
        f_scale=0.1,
    )
    return float(sol.x[1]) if sol.success else math.nan


def fixed_v_rms(x, tau, x0, t0, v):
    sol = least_squares(
        lambda p: hyperbola_time(x, p[0], p[1], v, 0.0) - tau,
        [x0, t0],
        loss="soft_l1",
        f_scale=0.1,
    )
    res = hyperbola_time(x, sol.x[0], sol.x[1], v, 0.0) - tau
    return float(np.sqrt(np.mean(res**2)))


def support_lines(sample: pd.DataFrame, cands: pd.DataFrame) -> list[int]:
    out = []
    for row in sample.itertuples():
        if pd.isna(row.d_cand_m):
            out.append(-1)
            continue
        same = cands[(cands["activity_id"] == row.activity_id) & (cands["path"] != row.path)]
        near = same[np.abs(same["d_cand_m"] - row.d_cand_m) <= SUPPORT_DEPTH_M]
        out.append(int(near["path"].nunique()))
    return out


def analyse_candidate(rg, inputs, row, config):
    v_meta = velocity_from_eps(rg.eps_r_meta)
    rec = {"v_meta": v_meta}
    if pd.isna(row.tau_ns):
        rec.update(curv_flag="unknown", unknown_reason="no_box", depth_status="no_box")
        return rec, None
    trace, sample = round(float(row.trace)), float(row.sample)
    box = candidate_box(rg, trace, sample, float(row.tau_ns))
    hint = ApexHint(trace=trace, sample=sample)
    result = estimate_depth(rg, box, config, inputs=inputs, apex_hint=hint)
    rec["depth_status"] = str(result.status)
    rec["depth_meta_m"] = result.depth_top_m
    picks = extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=rg.dt_ns,
        dx_m=rg.dx_m,
        f_c_hz=rg.f_c_hz,
        config=config,
        apex_hint=hint,
        period_px=inputs.branch.period_px,
    )
    rec["n_picks"] = picks.n_picks
    rec["pick_n_left"], rec["pick_n_right"] = picks.n_left, picks.n_right
    if picks.apex is None:
        rec.update(curv_flag="unknown", unknown_reason="no_apex")
        return rec, None
    rec["apex_trace"], rec["apex_sample"] = picks.apex.trace, picks.apex.k
    x, tau = picks.x_m, picks.tau_ns
    rec["aperture_m"] = float(np.max(np.abs(x - picks.apex_x_m)))
    # QC only: straight-ray fit deliberately run with the ground label.
    curv = fit_curvature(picks, radius_m=0.0, air_gap="ground", v_fallback=v_meta, config=config)
    if curv.fit is not None:
        fit = curv.fit
        rec.update(
            v_curv=fit.v_m_per_ns,
            sigma_v=fit.sigma_v,
            r=fit.v_m_per_ns / v_meta,
            rms_free_ns=fit.rms_ns,
            inlier_frac=fit.inlier_frac,
            at_bound=fit.at_bound,
            ransac_accepted=bool(curv.init.accepted) if curv.init else None,
            x0_fit_m=fit.x0_m,
            t0_fit_ns=fit.t0_ns,
        )
        rec["rms_meta_ns"] = fixed_v_rms(x, tau, fit.x0_m, fit.t0_ns, v_meta)
        left, right = x <= fit.x0_m, x >= fit.x0_m
        rec["v_left"] = side_velocity(x[left], tau[left], fit.x0_m, fit.t0_ns, v_meta)
        rec["v_right"] = side_velocity(x[right], tau[right], fit.x0_m, fit.t0_ns, v_meta)
    if curv.status == "insufficient_picks":
        rec.update(curv_flag="unknown", unknown_reason="insufficient_picks")
    elif curv.status != "ok":
        rec.update(curv_flag="unknown", unknown_reason="fit_failed")
    elif (
        not np.isfinite(rec["sigma_v"])
        or rec["sigma_v"] / v_meta > UNSTABLE_SIGMA_REL
        or rec["inlier_frac"] < UNSTABLE_INLIER
    ):
        rec.update(curv_flag="unknown", unknown_reason="unstable")
    else:
        rec.update(
            curv_flag="in" if abs(rec["r"] - 1.0) <= R_TOL + 1e-12 else "out",
            unknown_reason="",
        )
    return rec, (x, tau, curv)


def refraction_bias() -> pd.DataFrame:
    """Straight-ray R=0 fit of exact air-gap ray times; zero = surface reflection."""
    rows = []
    config = DepthConfig()
    for eps in (9.0, 16.0):
        v = velocity_from_eps(eps)
        for h in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
            for d in (0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
                for a in (0.3, 0.5, 0.8):
                    for radius in (0.0, 0.1):
                        x = np.arange(-a, a + 1e-9, 0.02)
                        t = reflection_times(
                            x, offset_m=0.0, h=h, x0_m=0.0, d_top_m=d, radius_m=radius, v=v
                        )
                        tau = t - surface_reflection_time(h, 0.0)
                        init = ransac_init(x, tau, 0.0, float(tau.min()), v, config)
                        fit = fit_hyperbola(
                            x, tau, radius_m=0.0, init=init, air_gap="ground", config=config
                        )
                        rows.append(
                            dict(
                                eps_r=eps,
                                h_m=h,
                                d_top_m=d,
                                half_aperture_m=a,
                                radius_m=radius,
                                r=fit.v_m_per_ns / v,
                                depth_straight_m=fit.v_m_per_ns * fit.t0_ns / 2,
                                rms_ns=fit.rms_ns,
                            )
                        )
    return pd.DataFrame(rows)


def ratios(df: pd.DataFrame, true_mask: pd.Series) -> dict:
    t, f = df[true_mask], df[~true_mask]

    def frac(s, cond):
        return None if len(s) == 0 else float(cond(s).mean())

    return {
        "n_true": int(len(t)),
        "n_false": int(len(f)),
        "true_kept_incl_unknown": frac(t, lambda s: s["curv_flag"] != "out"),
        "false_rejected": frac(f, lambda s: s["curv_flag"] == "out"),
        "true_in_among_known": frac(
            t[t["curv_flag"] != "unknown"], lambda s: s["curv_flag"] == "in"
        ),
        "false_out_among_known": frac(
            f[f["curv_flag"] != "unknown"], lambda s: s["curv_flag"] == "out"
        ),
        "unknown_true": frac(t, lambda s: s["curv_flag"] == "unknown"),
        "unknown_false": frac(f, lambda s: s["curv_flag"] == "unknown"),
    }


def best_band(df, true_mask, col, lo_grid, hi_grid):
    """Diagnostic only: Youden-best band on known r (not a recommendation)."""
    known = df[df[col].notna()]
    tm = true_mask[known.index]
    best = None
    for lo in lo_grid:
        for hi in hi_grid:
            if hi <= lo:
                continue
            inside = (known[col] >= lo) & (known[col] <= hi)
            tpr = float(inside[tm].mean()) if tm.any() else math.nan
            tnr = float((~inside[~tm]).mean()) if (~tm).any() else math.nan
            j = tpr + tnr - 1
            if best is None or j > best[0]:
                best = (j, lo, hi, tpr, tnr)
    return dict(zip(["youden", "lo", "hi", "tpr", "tnr"], best, strict=True))


def draw_example(ax, rg, inputs, row, rec, fitinfo):
    img = agc(inputs.branch.background_removed, rg.dt_ns, rg.f_c_hz)
    half = int(round(1.2 / rg.dx_m))
    i_c = int(round(row.trace))
    t_c = float(row.sample) * rg.dt_ns
    k0 = max(int((t_c - 8) / rg.dt_ns), 0)
    k1 = min(int((t_c + 16) / rg.dt_ns), img.shape[0] - 1)
    i0, i1 = max(i_c - half, 0), min(i_c + half, img.shape[1] - 1)
    sub = img[k0 : k1 + 1, i0 : i1 + 1]
    lim = float(np.percentile(np.abs(sub), 99))
    ext = ((i0 - i_c) * rg.dx_m, (i1 - i_c) * rg.dx_m, k1 * rg.dt_ns, k0 * rg.dt_ns)
    ax.imshow(sub, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=ext)
    tz = inputs.zero.t_zero_ns
    x_c = i_c * rg.dx_m
    if fitinfo is not None:
        x, tau, curv = fitinfo
        ax.plot(x - x_c, tau + tz, ".", color="#eda100", ms=3, label="picks")
        if curv.fit is not None:
            xx = np.linspace(x.min(), x.max(), 100)
            f = curv.fit
            ax.plot(
                xx - x_c,
                hyperbola_time(xx, f.x0_m, f.t0_ns, f.v_m_per_ns, 0.0) + tz,
                color="#2a78d6",
                lw=1.4,
                label="curvature fit",
            )
            vm = rec["v_meta"]
            ax.plot(
                xx - x_c,
                hyperbola_time(xx, f.x0_m, f.t0_ns, vm, 0.0) + tz,
                color="#eb6834",
                lw=1.2,
                ls="--",
                label="meta v",
            )
    ax.plot([0], [t_c - 2.5], marker="v", color="#e34948", ms=7)
    r = rec.get("r")
    rtxt = "unknown" if rec["curv_flag"] == "unknown" else f"r={r:.2f}"
    ax.set_title(f"{row.type} | {row.activity_id} {row.path} | {rtxt}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x - x_c (m)", fontsize=6)
    ax.set_ylabel("t (ns)", fontsize=6)


def main() -> None:
    config = DepthConfig()
    key = pd.read_csv(GALLERY / "sample_key.csv", dtype={"activity_id": str})
    review = pd.read_csv(GALLERY / "review_sheet_locked.csv")
    df = key.merge(review, on="blind_id", validate="one_to_one")
    cands = pd.read_csv(CANDIDATES, dtype={"activity_id": str})
    lines = pd.read_csv(DOCS / "fourtu_lines_v1.csv", dtype={"activity_id": str})
    eps = cands.merge(lines[["activity_id", "path", "eps_r_used"]], on=["activity_id", "path"], how="left")
    cands["d_cand_m"] = (
        eps["eps_r_used"].map(velocity_from_eps).to_numpy() * cands["tau_ns"].to_numpy() / 2
    )

    df["v_straight_qc"] = df["cand_index"].map(cands["v_straight_qc"])
    df["support_lines"] = support_lines(df, cands)

    # Global fixed bins (pooled tertiles of all auto candidate depths).
    auto_d = cands.loc[cands["source"] == "auto_v1", "d_cand_m"].dropna()
    g1, g2 = np.quantile(auto_d, [1 / 3, 2 / 3])
    df["fixed_bin"] = np.where(df["d_cand_m"].isna(), np.nan, np.digitize(df["d_cand_m"], [g1, g2]))

    recs, fits, store = [], {}, {}
    for (act, path), group in df.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg, config)
        for row in group.itertuples():
            rec, info = analyse_candidate(rg, inputs, row, config)
            rec["blind_id"] = row.blind_id
            recs.append(rec)
            fits[row.blind_id] = info
            store[row.blind_id] = (act, path)
    res = pd.DataFrame(recs)
    out = df.merge(res, on="blind_id", validate="one_to_one")
    out["asym_lr"] = np.abs(np.log(out["v_left"] / out["v_right"]))
    out["rms_ratio"] = out["rms_meta_ns"] / out["rms_free_ns"]
    out["moved"] = (np.abs(out["apex_trace"] - out["trace"].round()) > 3) | (
        np.abs(out["apex_sample"] - out["sample"]) > 0.5 * 2.0 / out["dt_ns"]
    )
    cols = [
        "blind_id", "activity_id", "site", "path", "source", "trace", "sample", "tau_ns",
        "d_cand_m", "depth_bin", "bin_q1_m", "bin_q2_m", "fixed_bin", "type", "confidence",
        "comment", "eps_r_used", "dt_ns", "zero_rule", "v_meta", "v_curv", "sigma_v", "r",
        "curv_flag", "unknown_reason", "n_picks", "pick_n_left", "pick_n_right", "aperture_m",
        "rms_free_ns", "rms_meta_ns", "rms_ratio", "inlier_frac", "v_left", "v_right",
        "asym_lr", "support_lines", "v_straight_qc", "depth_status", "moved",
    ]
    out = out[cols].sort_values(["site", "activity_id", "path", "trace"]).reset_index(drop=True)
    out.to_csv(DOCS / "m29c_apex_review.csv", index=False, float_format="%.4f")

    true_mask = out["type"].isin(TRUE_TYPES)
    summary = {
        "seed": 20261106,
        "n": int(len(out)),
        "fixed_bin_edges_m": [float(g1), float(g2)],
        "type_counts": out["type"].value_counts().to_dict(),
        "type_by_source": pd.crosstab(out["type"], out["source"]).to_dict(),
        "crosstab_type_flag": pd.crosstab(out["type"], out["curv_flag"]).to_dict(),
        "unknown_reasons": pd.crosstab(out["type"], out["unknown_reason"]).to_dict(),
        "rule_all": ratios(out, true_mask),
        "rule_auto": ratios(out[out["source"] == "auto_v1"], true_mask[out["source"] == "auto_v1"]),
        "rule_manual": ratios(
            out[out["source"] == "manual_v0"], true_mask[out["source"] == "manual_v0"]
        ),
        "rule_T_high_medium_vs_false": ratios(
            out[~(true_mask & (out["confidence"] == "low"))],
            true_mask[~(true_mask & (out["confidence"] == "low"))],
        ),
        "r_quantiles_by_type": {
            t: out.loc[(out["type"] == t) & out["r"].notna(), "r"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .round(3)
            .tolist()
            for t in TYPE_ORDER
        },
        "by_depth_bin": {
            int(b): ratios(out[out["depth_bin"] == b], true_mask[out["depth_bin"] == b])
            for b in (0, 1, 2)
        },
        "by_fixed_bin": {
            int(b): ratios(out[out["fixed_bin"] == b], true_mask[out["fixed_bin"] == b])
            for b in (0, 1, 2)
        },
        "true_frac_by_depth_bin": out.groupby("depth_bin")["type"]
        .apply(lambda s: float(s.isin(TRUE_TYPES).mean()))
        .to_dict(),
        "by_site": {
            int(s): {
                "n": int(len(g)),
                "n_true": int(g["type"].isin(TRUE_TYPES).sum()),
                "flags": g["curv_flag"].value_counts().to_dict(),
                "r_median_true": float(g.loc[g["type"].isin(TRUE_TYPES), "r"].median())
                if g["type"].isin(TRUE_TYPES).any()
                else None,
            }
            for s, g in out.groupby("site")
        },
        "best_band_r_diagnostic": best_band(
            out, true_mask, "r", np.arange(0.5, 1.2, 0.05), np.arange(1.0, 3.01, 0.05)
        ),
        "metric_medians_true_vs_false": {
            m: [
                float(out.loc[true_mask, m].median()),
                float(out.loc[~true_mask, m].median()),
            ]
            for m in ["rms_free_ns", "rms_meta_ns", "rms_ratio", "asym_lr", "aperture_m", "support_lines", "n_picks"]
        },
    }
    # Alternative single-metric rules (diagnostic, thresholds not tuned on truth labels
    # beyond reporting): computed as keep/reject rates on the review labels.
    alts = {}
    known = out
    for name, cond in {
        "r_in_0.7_1.3": (known["curv_flag"] != "out"),
        "rms_meta<=0.25ns": known["rms_meta_ns"].fillna(np.inf) <= 0.25,
        "asym_lr<=0.4": known["asym_lr"].fillna(np.inf) <= 0.4,
        "support>=1": known["support_lines"] >= 1,
        "r_in_and_asym": (known["curv_flag"] == "in") & (known["asym_lr"].fillna(np.inf) <= 0.4),
        "r_0.7_2.0": known["r"].between(0.7, 2.0) | known["r"].isna(),
    }.items():
        alts[name] = {
            "true_kept": float(cond[true_mask].mean()),
            "false_rejected": float((~cond[~true_mask]).mean()),
        }
    summary["alternative_rules_diagnostic"] = alts

    bias = refraction_bias()
    bias.to_csv(DOCS / "m29c_refraction_bias.csv", index=False, float_format="%.4f")
    piv = bias[(bias["radius_m"] == 0.0) & (bias["eps_r"] == 9.0)].pivot_table(
        index=["h_m", "half_aperture_m"], columns="d_top_m", values="r"
    )
    summary["refraction_r_eps9_R0"] = {str(k): v for k, v in piv.round(3).to_dict("index").items()}
    (DOCS / "m29c_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )

    # Figure 1: r by type (strip), +-30 % band shaded.
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.axhspan(0.7, 1.3, color="#d9d9d9", alpha=0.6, lw=0)
    rng = np.random.default_rng(0)
    xt = []
    for i, t in enumerate(TYPE_ORDER):
        sub = out[(out["type"] == t)]
        r = sub["r"].where(sub["curv_flag"] != "unknown")
        rr = r.dropna().clip(upper=3.0)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(rr)), rr, s=18, color="#2a78d6", lw=0)
        nu = int((sub["curv_flag"] == "unknown").sum())
        xt.append(f"{t}\n{TYPE_NAMES[t]}\nn={len(sub)} (unk {nu})")
    ax.set_xticks(range(len(TYPE_ORDER)), xt, fontsize=6.5)
    ax.set_ylabel("r = v_curv / v_meta (clipped at 3)")
    ax.set_title("Curvature-v ratio by reviewed type (grey: +-30 % band)", fontsize=10)
    ax.grid(axis="y", color="#eeeeee")
    fig.savefig(DOCS / "m29c_r_by_type.png", dpi=130)
    plt.close(fig)

    # Figure 2: representative montage (two per type, high/medium confidence first).
    picks_rows = []
    for t in ["T", "T", "W", "L", "L", "C", "X", "A", "B"]:
        pool = out[(out["type"] == t) & ~out["blind_id"].isin([p.blind_id for p in picks_rows])]
        pool = pool.assign(cr=pool["confidence"].map({"high": 0, "medium": 1, "low": 2}))
        pool = pool.sort_values(["cr", "blind_id"])
        if len(pool):
            picks_rows.append(pool.iloc[0])
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    loaded = {}
    for ax, row in zip(axes.ravel(), picks_rows, strict=False):
        k = (row.activity_id, row.path)
        if k not in loaded:
            rg = load_fourtu_radargram(BASE, *k)
            loaded[k] = (rg, prepare_depth_inputs(rg, config))
        rg, inputs = loaded[k]
        rec = res.set_index("blind_id").loc[row.blind_id].to_dict()
        draw_example(ax, rg, inputs, SimpleNamespace(**row.to_dict()), rec, fits[row.blind_id])
    for ax in axes.ravel()[len(picks_rows):]:
        ax.axis("off")
    axes.ravel()[1].legend(fontsize=6, loc="lower left")
    fig.suptitle("M2-9 C reviewed apex-candidate types (4TU, CC0 1.0): picks, curvature fit, meta-v hyperbola", fontsize=10)
    fig.savefig(DOCS / "m29c_types_montage.png", dpi=110)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in ["n", "type_counts", "crosstab_type_flag", "rule_all"]}, default=str))


if __name__ == "__main__":
    main()
