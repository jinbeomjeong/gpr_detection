"""E3x plan-structure-fixed MDE evaluation (stats-design-expert, synthetic outcomes only).

The position / cluster / model / background structure is read from the sealed plan tables
(runs/eval/m33/e3x/plan/hyperbola_positions.csv, band_positions.csv; detection-free).
Outcomes are synthetic, from the same latent-threshold model as the [STV] review simulator
(scratchpad stats-e3x/e3x_sim.py), calibrated on public Stage C R1 r0 data.

Rules re-implemented from docs/processing/m33_e3x_plan.md v0.4 sections 3-2..3-4:
  * per-hypothesis stratified cluster bootstrap over the clusters that hold target positions,
  * Satterthwaite t interval: v_g = G/(G-1) Var_b(partial replicate g), nu_g = max(G_eff-1, 1),
    nu = (sum v)^2 / sum(v^2/nu_g), level exactly 1 - 0.05/6 (one tail 0.05/12),
  * model-equal weighting (HH1, negative control, HH3), background-equal weighting (HB*),
  * models missing in a replicate are dropped and weights re-normalised.
Power uses the shift trick: Satterthwaite bounds are shift-equivariant in Delta.

Usage: e3x_plan_mde.py <config.json> <out.json>
stdout: one status line.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as tdist

LEVEL = 1.0 - 0.05 / 6.0
TAIL = 0.05 / 12.0
DELTA = 0.05

# Stage C v2 r0 per-model mean of Y (12/18 dB, 5 folds, both polarities), main cell
# (same prior as the [STV] simulator, source runs/eval/m32_recipe_v2/injected/r0/fold*/inj_matches.csv).
STAGEC_MU = {
    "M01": 0.700, "M02": 0.454, "M03": 0.217, "M04": 0.071, "M05": 0.521, "M06": 0.500,
    "M07": 0.414, "M08": 0.276, "M09": 0.057, "M10": 0.628, "M11": 0.486, "M12": 0.174,
    "M14": 0.940, "M15": 0.000, "M16": 0.044, "M17": 0.941, "M18": 0.157, "M19": 0.574,
}
MANIP_MODELS_BY_PRIOR = ["M15", "M16", "M09", "M04", "M18", "M03", "M08"]  # ascending Stage C mean


def load_plan(plan_dir: Path):
    h = pd.read_csv(plan_dir / "hyperbola_positions.csv", dtype={"cluster_id": str, "edge_snr_set": str, "line_id": str})
    h["bg"] = np.where(h["dataset"] == "4tu", "4tu", "ids")
    h["cluster"] = h["bg"] + ":" + h["cluster_id"].astype(str)
    h["s_edge"] = [[int(x) for x in str(s).split(",")] if isinstance(s, str) and s.strip() else [] for s in h["edge_snr_set"]]
    b = pd.read_csv(plan_dir / "band_positions.csv", dtype={"cluster_id": str, "line_id": str})
    b["bg"] = np.where(b["dataset"] == "4tu", "4tu", "ids")
    b["cluster"] = b["bg"] + ":" + b["cluster_id"].astype(str)
    b["has90"] = b["has_deep_row"].astype(str).str.lower().eq("true")
    return h.reset_index(drop=True), b.reset_index(drop=True)


# ---- data generation (same model as [STV]) ------------------------------------------
def q_for(mu, tau):
    mu = np.clip(mu, 0.002, 0.998)
    return np.sqrt(1 + tau**2) * norm.ppf(mu)


def gen_units(rng, a, n_arms, rho, s_w2, n_folds=5, efold=None, rho_f=0.0):
    P = a.shape[0]
    s_e2 = 1.0 - s_w2
    wc = rng.standard_normal((P, 1))
    wi = rng.standard_normal((P, n_arms))
    w = np.sqrt(s_w2) * (np.sqrt(rho) * wc + np.sqrt(1 - rho) * wi)
    ec = rng.standard_normal((P, 1, n_folds))
    ei = rng.standard_normal((P, n_arms, n_folds))
    e_loc = np.sqrt(rho) * ec + np.sqrt(1 - rho) * ei
    if efold is not None and rho_f > 0:
        e_loc = np.sqrt(rho_f) * efold[:, None, :] + np.sqrt(1 - rho_f) * e_loc
    e = np.sqrt(s_e2) * e_loc
    aa = a[:, None, None] if a.ndim == 1 else a[:, :, None]
    return ((aa + w[:, :, None] + e) > 0).mean(axis=2)


def position_latent(rng, clusters, base_q, icc_by_bg, bg, tau):
    uc = {}
    out = np.empty(len(clusters))
    for i, (c, g) in enumerate(zip(clusters, bg)):
        if c not in uc:
            uc[c] = rng.standard_normal() * np.sqrt(icc_by_bg[g]) * tau
        out[i] = base_q[i] + uc[c] + rng.standard_normal() * np.sqrt(1 - icc_by_bg[g]) * tau
    return out


# ---- estimator with per-hypothesis stratified cluster bootstrap ---------------------
def estimate(rng, B, clusters, bgs, groups, values, drop_small_bg=None):
    """Equal-weight mean over groups of within-group position means + Satterthwaite interval."""
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return None
    groups = list(groups)
    if drop_small_bg is not None:
        cnt = pd.Series(bgs).value_counts()
        keep_bg = [g for g in ("4tu", "ids") if cnt.get(g, 0) >= drop_small_bg]
        if not keep_bg:
            keep_bg = [g for g in ("4tu", "ids") if cnt.get(g, 0) > 0]
        k = np.array([g in keep_bg for g in bgs])
        clusters = [c for c, kk in zip(clusters, k) if kk]
        bgs = [c for c, kk in zip(bgs, k) if kk]
        groups = [c for c, kk in zip(groups, k) if kk]
        values = values[k]
    glist = sorted(set(groups))
    gi = {g: j for j, g in enumerate(glist)}
    strata = {}
    for c, g in zip(clusters, bgs):
        strata.setdefault(g, [])
        if c not in strata[g]:
            strata[g].append(c)
    cols = [(g, c) for g in sorted(strata) for c in strata[g]]
    ci = {c: j for j, (_, c) in enumerate(cols)}
    C = len(cols)
    S = np.zeros((C, len(glist)))
    N = np.zeros((C, len(glist)))
    for c, g, v in zip(clusters, groups, values):
        S[ci[c], gi[g]] += v
        N[ci[c], gi[g]] += 1

    def ev(W):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (W @ S) / (W @ N)
        return np.nanmean(m, axis=1)

    full = float(ev(np.ones((1, C)))[0])
    K = np.ones((B, C))
    idx_by = {}
    for g in sorted(strata):
        idx = np.array([ci[c] for c in strata[g]])
        idx_by[g] = idx
        K[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=B)
    with np.errstate(invalid="ignore"):
        bs = ev(K)
    vs, dfs, flags = [], [], []
    G_by, Geff_by = {}, {}
    for g, idx in idx_by.items():
        n_c = N[idx].sum(axis=1)
        G = len(idx)
        G_by[g] = G
        Geff_by[g] = float(n_c.sum() ** 2 / np.sum(n_c**2))
        if G < 2:
            flags.append("single_cluster_background")
            continue
        W = np.ones((B, C))
        W[:, idx] = K[:, idx]
        v = float(np.var(ev(W), ddof=1)) * G / (G - 1)
        if v <= 0:
            flags.append("zero_variance_background")
            continue
        vs.append(v)
        dfs.append(max(Geff_by[g] - 1.0, 1.0))
    V = float(sum(vs))
    if V > 0:
        nu = V**2 / sum(v**2 / d for v, d in zip(vs, dfs))
        q = float(tdist.ppf(1 - TAIL, nu))
        lo, hi = full - q * np.sqrt(V), full + q * np.sqrt(V)
    else:
        nu, lo, hi = float("nan"), full, full
        flags.append("degenerate_ci")
    plo, phi = np.quantile(bs, [TAIL, 1 - TAIL])
    return {"est": full, "lo": float(lo), "hi": float(hi), "nu": float(nu), "se": float(np.sqrt(V)),
            "plo": float(plo), "phi": float(phi), "n_pos": int(len(values)), "n_groups": len(glist),
            "G": G_by, "G_eff": Geff_by, "n_clu": int(C), "flags": flags}


# ---- one replicate ------------------------------------------------------------------
def replicate(rng, cfg, H, Bd, mstar_sets):
    tau_h, s_w2 = cfg["tau_h"], cfg["s_w2"]
    icc_h = {"4tu": cfg["icc_h_4tu"], "ids": cfg["icc_h_ids"]}
    models = list(H["model"])
    bg = list(H["bg"])
    hcl = list(H["cluster"])
    P = len(H)
    mu = np.array([STAGEC_MU[m] for m in models])
    qshift = cfg.get("q_shift", 0.0) + np.where(np.array(bg) == "ids", cfg.get("q_shift_ids", 0.0), 0.0)
    a = position_latent(rng, hcl, q_for(mu, tau_h) + qshift, icc_h, bg, tau_h)
    rho_f = cfg.get("rho_f", 0.0)
    ef = rng.standard_normal((P, 5))
    is_nc = (H["edge_stratum"] == "negative_control").to_numpy()
    rho = np.where(is_nc, cfg["rho_nc"], cfg["rho_man"])
    sd_cd = cfg.get("sd_cluster_diff", 0.0)
    ucl = sorted(set(hcl))
    g_on = {c: rng.standard_normal() * sd_cd for c in ucl}
    g_30 = {c: rng.standard_normal() * sd_cd for c in ucl}
    off_on = np.array([0.0 if nc else g_on[c] for c, nc in zip(hcl, is_nc)])
    off_30 = np.array([g_30[c] for c in hcl])
    a2 = np.stack([a + off_on, a], axis=1)
    y = {}
    for s in (12, 18):
        for pol in (1, -1):
            yy = np.empty((P, 2))
            for r in np.unique(rho):
                sel = rho == r
                yy[sel] = gen_units(rng, a2[sel], 2, r, s_w2, efold=ef[sel], rho_f=rho_f)
            y[("on", s, pol)] = yy[:, 0]
            y[("off", s, pol)] = yy[:, 1]
    for pol in (1, -1):
        y[("on", 30, pol)] = gen_units(rng, a + off_30, 1, 0.0, s_w2, efold=ef, rho_f=rho_f)[:, 0]
    Y_on = np.mean([y[("on", s, p)] for s in (12, 18) for p in (1, -1)], axis=0)
    Y_off = np.mean([y[("off", s, p)] for s in (12, 18) for p in (1, -1)], axis=0)
    mean2 = pd.Series((Y_on + Y_off) / 2).groupby(np.array(models)).mean()
    mstar_nat = set(mean2[mean2 >= 0.10].index)
    y12 = np.mean([y[("on", 12, p)] for p in (1, -1)], axis=0)
    y30 = np.mean([y[("on", 30, p)] for p in (1, -1)], axis=0)
    d_hh1 = np.array([np.mean([y[("on", s, p)][i] - y[("off", s, p)][i] for s in se for p in (1, -1)]) if se else np.nan
                      for i, se in enumerate(H["s_edge"])])
    d_full = Y_on - Y_off
    d_hh3 = y12 - y30
    manip = (H["edge_stratum"] == "manipulated").to_numpy()
    B = cfg["B"]
    out = {"hyp": {}}
    for tag, ms in [("natural", mstar_nat)] + list(mstar_sets.items()):
        in_m = np.array([m in ms for m in models])
        rec = {}
        sel = manip & in_m
        rec["HH1"] = estimate(rng, B, [hcl[i] for i in np.where(sel)[0]], [bg[i] for i in np.where(sel)[0]],
                              [models[i] for i in np.where(sel)[0]], d_hh1[sel])
        if rec["HH1"] is not None:
            rec["HH1"]["n_models"] = len({models[i] for i in np.where(sel)[0]})
        sel = is_nc & in_m
        rec["NC"] = estimate(rng, B, [hcl[i] for i in np.where(sel)[0]], [bg[i] for i in np.where(sel)[0]],
                             [models[i] for i in np.where(sel)[0]], d_full[sel])
        sel = in_m
        rec["HH3"] = estimate(rng, B, [hcl[i] for i in np.where(sel)[0]], [bg[i] for i in np.where(sel)[0]],
                              [models[i] for i in np.where(sel)[0]], d_hh3[sel])
        if tag == "natural":
            rec["HH1_full"] = estimate(rng, B, [hcl[i] for i in np.where(sel)[0]], [bg[i] for i in np.where(sel)[0]],
                                       [models[i] for i in np.where(sel)[0]], d_full[sel])
            rec["mstar"] = sorted(ms)
        out["hyp"][tag] = rec
    # bands
    tau_b = cfg["tau_b"]
    icc_b = {"4tu": cfg["icc_b_4tu"], "ids": cfg["icc_b_ids"]}
    bbg = list(Bd["bg"])
    bcl = list(Bd["cluster"])
    mu_b = np.where(np.array(bbg) == "ids", cfg["band_mu_ids"], cfg["band_mu_4tu"])
    ab = position_latent(rng, bcl, q_for(mu_b, tau_b), icc_b, bbg, tau_b)
    PB = len(Bd)
    ef_b = rng.standard_normal((PB, 5))
    arms = ["28", "47", "68", "90", "47s", "68s"]
    gb = {c: rng.standard_normal(len(arms)) * sd_cd for c in sorted(set(bcl))}
    ab2 = ab[:, None] + np.array([gb[c] for c in bcl])
    yb = {k: np.zeros(PB) for k in arms}
    for s in (12, 18):
        for pol in (1, -1):
            u = gen_units(rng, ab2, len(arms), cfg.get("rho_band", 0.0), cfg["s_w2_b"], efold=ef_b,
                          rho_f=cfg.get("rho_f_b", rho_f))
            for j, k in enumerate(arms):
                yb[k] += u[:, j] / 4
    d_hb1 = (yb["47"] + yb["68"]) / 2 - yb["28"]
    d_hb1b = (yb["47"] + yb["68"]) / 2 - yb["90"]
    d_hb2 = ((yb["47"] - yb["47s"]) + (yb["68"] - yb["68s"])) / 2
    h90 = Bd["has90"].to_numpy()
    out["HB1"] = estimate(rng, B, bcl, bbg, bbg, d_hb1, drop_small_bg=15)
    out["HB1b"] = estimate(rng, B, [bcl[i] for i in np.where(h90)[0]], [bbg[i] for i in np.where(h90)[0]],
                           [bbg[i] for i in np.where(h90)[0]], d_hb1b[h90], drop_small_bg=15)
    out["HB2"] = estimate(rng, B, bcl, bbg, bbg, d_hb2, drop_small_bg=15)
    return out


# ---- summaries ----------------------------------------------------------------------
def q_mcse(x, p, rng, nb=400):
    x = np.asarray(x)
    qs = [np.quantile(x[rng.integers(0, len(x), len(x))], p) for _ in range(nb)]
    return float(np.std(qs, ddof=1))


def summarize_rec(rr, rng):
    rr = [r for r in rr if r is not None]
    R = len(rr)
    if R == 0:
        return None
    est = np.array([r["est"] for r in rr])
    lo = np.array([r["lo"] for r in rr])
    hi = np.array([r["hi"] for r in rr])
    plo = np.array([r["plo"] for r in rr])
    phi = np.array([r["phi"] for r in rr])
    fa = (lo > 0) | (hi < 0)
    d = {"R": R,
         "n_pos_mean": float(np.mean([r["n_pos"] for r in rr])),
         "n_pos_min": int(np.min([r["n_pos"] for r in rr])),
         "n_clu_mean": float(np.mean([r["n_clu"] for r in rr])),
         "n_clu_min": int(np.min([r["n_clu"] for r in rr])),
         "nu_median": float(np.nanmedian([r["nu"] for r in rr])),
         "halfwidth_mean": float(np.mean((hi - lo) / 2)),
         "sd_est_true": float(np.std(est, ddof=1)),
         "boot_se_over_true_sd": float(np.mean([r["se"] for r in rr]) / np.std(est, ddof=1)) if np.std(est) > 0 else None,
         "false_claim_two_sided_at0": float(np.mean(fa)),
         "false_claim_mcse": float(np.sqrt(max(np.mean(fa) * (1 - np.mean(fa)), 1e-12) / R)),
         "p_lo_gt0": float(np.mean(lo > 0)), "p_hi_lt0": float(np.mean(hi < 0)),
         "mde80": float(np.quantile(-lo, 0.80)),
         "mde80_mcse": q_mcse(-lo, 0.80, rng),
         "mde50": float(np.quantile(-lo, 0.50)),
         "mde80_neg": float(np.quantile(hi, 0.80)),
         "mde80_percentile_report_only": float(np.quantile(-plo, 0.80)),
         "pct_false_claim_at0_report_only": float(np.mean((plo > 0) | (phi < 0))),
         "p_negligible_at0": float(np.mean((lo > -DELTA) & (hi < DELTA))),
         "p_false_negligible_at_delta": float(np.mean(hi + DELTA < DELTA)),
         "flags": pd.Series([f for r in rr for f in r["flags"]]).value_counts().to_dict()}
    curve = {}
    for dl in (0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15):
        curve[f"{dl:.2f}"] = float(np.mean(lo + dl > 0))
    d["power_curve_confirm"] = curve
    if "n_models" in rr[0]:
        nm = np.array([r["n_models"] for r in rr])
        d["n_models_mean"] = float(nm.mean())
        d["n_models_dist"] = {int(k): int(v) for k, v in zip(*np.unique(nm, return_counts=True))}
    return d


def nc_trigger(rr):
    rr = [r for r in rr if r is not None]
    out = {}
    for dl in (0.0, 0.02, 0.03, 0.05, 0.07, 0.10):
        a = [((r["lo"] + dl > 0) or (r["hi"] + dl < 0)) and abs(r["est"] + dl) >= 0.05 for r in rr]
        b = [((r["lo"] + dl > 0) or (r["hi"] + dl < 0)) and 0.02 <= abs(r["est"] + dl) < 0.05 for r in rr]
        out[f"{dl:.2f}"] = {"not_evaluable_a": float(np.mean(a)), "flag_b": float(np.mean(b))}
    return out


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outp = Path(sys.argv[2])
    H, Bd = load_plan(Path(cfg["plan_dir"]))
    rng = np.random.default_rng(cfg.get("seed", 20261114))
    mstar_sets = {}
    all_models = sorted(STAGEC_MU)
    for k in cfg.get("forced_exclusions", []):
        excl = set(MANIP_MODELS_BY_PRIOR[:k])
        mstar_sets[f"Me{14 - k}"] = set(all_models) - excl
    for name, excl in cfg.get("forced_named", {}).items():
        mstar_sets[name] = set(all_models) - set(excl)
    t0 = time.time()
    reps = [replicate(rng, cfg, H, Bd, mstar_sets) for _ in range(cfg["R"])]
    srng = np.random.default_rng(7)
    res = {"config": cfg, "elapsed_s": None, "structure": {
        "hyp_positions": int(len(H)), "band_positions": int(len(Bd)),
        "hb1b_positions": int(Bd["has90"].sum())}}
    tags = ["natural"] + list(mstar_sets)
    res["hyp"] = {}
    for tag in tags:
        rt = {}
        for h in ("HH1", "NC", "HH3", "HH1_full"):
            if tag != "natural" and h == "HH1_full":
                continue
            rt[h] = summarize_rec([r["hyp"][tag][h] for r in reps], srng)
        rt["NC_trigger"] = nc_trigger([r["hyp"][tag]["NC"] for r in reps])
        if tag != "natural":
            rt["mstar"] = sorted(mstar_sets[tag])
        res["hyp"][tag] = rt
    # natural M* diagnostics and minimum-count risk
    nat = [r["hyp"]["natural"] for r in reps]
    nm = np.array([x["HH1"]["n_models"] if x["HH1"] else 0 for x in nat])
    npos = np.array([x["HH1"]["n_pos"] if x["HH1"] else 0 for x in nat])
    nclu = np.array([x["HH1"]["n_clu"] if x["HH1"] else 0 for x in nat])
    hh1_ne = (nm < 5) | (npos < 30) | (nclu < 15)
    ncp = np.array([x["NC"]["n_pos"] if x["NC"] else 0 for x in nat])
    ncc = np.array([x["NC"]["n_clu"] if x["NC"] else 0 for x in nat])
    ncg = np.array([min(x["NC"]["G"].values()) if x["NC"] else 0 for x in nat])
    nc_nr = (ncp < 10) | (ncc < 5) | (ncg < 2)
    h3p = np.array([x["HH3"]["n_pos"] if x["HH3"] else 0 for x in nat])
    h3c = np.array([x["HH3"]["n_clu"] if x["HH3"] else 0 for x in nat])
    res["natural_mstar"] = {
        "p_hh1_not_evaluable_counts": float(hh1_ne.mean()),
        "p_nc_not_run_counts": float(nc_nr.mean()),
        "p_hh3_not_evaluable_counts": float(((h3p < 40) | (h3c < 20)).mean()),
        "hh1_models_dist": {int(k): int(v) for k, v in zip(*np.unique(nm, return_counts=True))},
        "hh1_pos_q05_q50": [float(np.quantile(npos, 0.05)), float(np.median(npos))],
        "nc_pos_min_median": [int(ncp.min()), float(np.median(ncp))],
        "mstar_inclusion_freq": pd.Series([m for x in nat for m in x["mstar"]]).value_counts().div(len(nat)).round(4).to_dict(),
    }
    for h in ("HB1", "HB1b", "HB2"):
        res[h] = summarize_rec([r[h] for r in reps], srng)
    fam = ["HH1", "HH3", "HB1", "HB1b", "HB2"]
    jf = []
    for r in reps:
        recs = [r["hyp"]["natural"]["HH1"], r["hyp"]["natural"]["HH3"], r["HB1"], r["HB1b"], r["HB2"]]
        jf.append(any(x is not None and (x["lo"] > 0 or x["hi"] < 0) for x in recs))
    res["joint_false_claim_5_linear_null"] = float(np.mean(jf))
    res["joint_false_claim_mcse"] = float(np.sqrt(np.mean(jf) * (1 - np.mean(jf)) / len(jf)))
    res["family"] = fam
    res["elapsed_s"] = round(time.time() - t0, 1)
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(outp), "R": cfg["R"], "B": cfg["B"], "elapsed_s": res["elapsed_s"]}))


def enumerate_minimums(plan_dir: Path, outp: Path):
    """Deterministic count check over every M* subset (no outcomes)."""
    H, _ = load_plan(plan_dir)
    manip = H[H["edge_stratum"] == "manipulated"]
    mm = sorted(manip["model"].unique())
    res = {"manip_models": mm, "by_size": {}}
    for k in range(0, len(mm) + 1):
        worst = None
        n_fail = 0
        n_all = 0
        for sub in itertools.combinations(mm, k):
            d = manip[manip["model"].isin(sub)]
            npos, nclu = len(d), d["cluster"].nunique()
            gg = d.groupby("bg")["cluster"].nunique().to_dict()
            fail = (k < 5) or (npos < 30) or (nclu < 15) or any(v < 2 for v in gg.values())
            n_all += 1
            n_fail += fail
            key = (npos, nclu)
            if worst is None or key < worst[0]:
                worst = (key, list(sub), gg)
        res["by_size"][k] = {"subsets": n_all, "fail": n_fail, "min_pos": worst[0][0], "min_clu": worst[0][1],
                             "argmin": worst[1], "argmin_G": worst[2]}
    nc = H[H["edge_stratum"] == "negative_control"]
    ncm = sorted(nc["model"].unique())
    res["nc_models"] = ncm
    res["nc_subsets"] = []
    for k in range(0, len(ncm) + 1):
        for sub in itertools.combinations(ncm, k):
            d = nc[nc["model"].isin(sub)]
            gg = d.groupby("bg")["cluster"].nunique().to_dict()
            nr = (len(d) < 10) or (d["cluster"].nunique() < 5) or any(v < 2 for v in gg.values())
            res["nc_subsets"].append({"models": list(sub), "pos": int(len(d)), "clu": int(d["cluster"].nunique()),
                                      "G": gg, "geff": {g: float(x.groupby("cluster").size().sum() ** 2 / (x.groupby("cluster").size() ** 2).sum())
                                                        for g, x in d.groupby("bg")}, "not_run": bool(nr)})
    # forced-set counts for HH1 / HH3
    res["forced_counts"] = {}
    all_models = sorted(STAGEC_MU)
    for k in range(0, 8):
        ms = set(all_models) - set(MANIP_MODELS_BY_PRIOR[:k])
        d = manip[manip["model"].isin(ms)]
        a = H[H["model"].isin(ms)]
        res["forced_counts"][f"Me{14 - k}"] = {
            "excluded": MANIP_MODELS_BY_PRIOR[:k],
            "hh1_pos": int(len(d)), "hh1_clu": int(d["cluster"].nunique()),
            "hh1_G": d.groupby("bg")["cluster"].nunique().to_dict(),
            "hh1_geff": {g: float(x.groupby("cluster").size().sum() ** 2 / (x.groupby("cluster").size() ** 2).sum()) for g, x in d.groupby("bg")},
            "hh1_18only_share": float(np.mean([s == [18] for s in d["s_edge"]])),
            "hh3_pos": int(len(a)), "hh3_clu": int(a["cluster"].nunique())}
    outp.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(outp)}))


if __name__ == "__main__":
    if sys.argv[1] == "--enumerate":
        enumerate_minimums(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
