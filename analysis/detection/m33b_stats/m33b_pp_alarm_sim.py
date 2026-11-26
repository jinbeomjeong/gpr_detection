"""M3-3b GPR-SD PP alarm operating characteristics and CI coverage (design-only).

No detection output and no labels are read. The unit structure (4 roads with
6/5/4/1 window triplets, 3 parallel lines, 5 folds) comes from the plan text
(docs/experiments/m33b_gprsd_plan.md section 6). GT count n is a parameter.

Generative model (latent threshold, per GT = pipe j in line l, fold f):
  z = a_road + b_pipe + c_pipe_line + d_pipe_fold + e
  S0 match  <=>  z > tau  (tau set so that marginal P = R0)
PP loss: pipe-level susceptibility (road + pipe latent > kappa, P = pi_loss);
  a susceptible matched GT-fold is lost with prob h.
PP gain (PP-b clip raising IoU): pipe-level susceptibility (P = pi_gain); a
  susceptible unmatched near-miss GT-fold (tau - band*sd < z <= tau) becomes
  matched with prob hg.
True effect delta = E[dR] = -R0*pi_loss*h + P_near*pi_gain*hg.
"""

import json
import sys
import time

import numpy as np
from scipy import stats

SEED = 20261114
ROAD_TRIPLETS = [6, 5, 4, 1]  # AR1, AR2, CR1, CR2
N_FOLDS = 5
N_LINES = 3
Q_LAB = 0.95  # probability a pipe of a triplet is labelled in a given line

VAR_SCEN = {
    # a_road, b_pipe, c_pipe_line, d_pipe_fold, e_resid
    "base": (0.5, 1.5, 0.1, 0.3, 0.1),
    "weak_road": (0.1, 1.9, 0.1, 0.3, 0.1),
    "strong_road": (1.0, 1.0, 0.1, 0.3, 0.1),
    "loose_lines": (0.5, 1.0, 0.6, 0.3, 0.1),
}
SUS_ROAD_VAR = 0.5  # road share of PP susceptibility latent (zero offset is per line/road)
SUS_PIPE_VAR = 0.5


def make_design(n_target, rng):
    t_count = sum(ROAD_TRIPLETS)
    road_of_t = np.repeat(np.arange(4), ROAD_TRIPLETS)
    n_pipes = max(t_count, int(round(n_target / (N_LINES * Q_LAB))))
    alloc = np.ones(t_count, int) + rng.multinomial(n_pipes - t_count, np.ones(t_count) / t_count)
    pipe_t = np.repeat(np.arange(t_count), alloc)
    labeled = rng.random((n_pipes, N_LINES)) < Q_LAB
    none = ~labeled.any(axis=1)
    labeled[none, 0] = True
    return {
        "pipe_t": pipe_t,
        "pipe_r": road_of_t[pipe_t],
        "labeled": labeled,
        "n": int(labeled.sum()),
        "n_pipes": int(n_pipes),
        "alloc": alloc.tolist(),
    }


def draw_z(design, var, reps, rng):
    sa, sb, sc, sd, se = np.sqrt(var)
    n_p = design["n_pipes"]
    pr = design["pipe_r"]
    a = rng.normal(size=(reps, 4)) * sa
    b = rng.normal(size=(reps, n_p)) * sb
    c = rng.normal(size=(reps, n_p, N_LINES)) * sc
    d = rng.normal(size=(reps, n_p, N_FOLDS)) * sd
    e = rng.normal(size=(reps, n_p, N_LINES, N_FOLDS)) * se
    z = a[:, pr][:, :, None, None] + b[:, :, None, None] + c[:, :, :, None] + d[:, :, None, :] + e
    return z, a


def pipe_susceptible(design, p, reps, rng):
    n_p = design["n_pipes"]
    if p <= 0:
        return np.zeros((reps, n_p), bool)
    if p >= 1:
        return np.ones((reps, n_p), bool)
    kappa = stats.norm.ppf(1 - p)
    g = rng.normal(size=(reps, 4)) * np.sqrt(SUS_ROAD_VAR)
    w = rng.normal(size=(reps, n_p)) * np.sqrt(SUS_PIPE_VAR)
    return (g[:, design["pipe_r"]] + w) > kappa


def simulate_alarm(design, var, r0, pi_loss, h, pi_gain, hg, band, reps, rng):
    tot = float(np.sqrt(sum(var)))
    tau = -stats.norm.ppf(r0) * tot
    lab = design["labeled"][None, :, :, None]
    z, _ = draw_z(design, var, reps, rng)
    m0 = (z > tau) & lab
    sus = pipe_susceptible(design, pi_loss, reps, rng)
    lost = m0 & sus[:, :, None, None] & (rng.random(m0.shape) < h)
    gsus = pipe_susceptible(design, pi_gain, reps, rng)
    near = (z <= tau) & (z > tau - band * tot) & lab
    gained = near & gsus[:, :, None, None] & (rng.random(m0.shape) < hg)
    m3 = (m0 & ~lost) | gained
    n5 = design["n"] * N_FOLDS
    big_m0 = m0.sum(axis=(1, 2, 3))
    big_m3 = m3.sum(axis=(1, 2, 3))
    d_r = (big_m3 - big_m0) / n5
    # distinct triplets with any S0 match
    t_idx = design["pipe_t"]
    any_t = np.zeros((reps, 16), bool)
    m0_pipe = m0.any(axis=(2, 3))
    for t in range(16):
        any_t[:, t] = m0_pipe[:, t_idx == t].any(axis=1)
    n_trip_matched = any_t.sum(axis=1)
    # distinct GT (pipe-line) with S0 match in any fold
    n_gt_matched = m0.any(axis=3).sum(axis=(1, 2))
    # GT lost (net) in >= 3 of 5 folds
    net_lost_gf = m0 & ~m3
    gt_lost3 = (net_lost_gf.sum(axis=3) >= 3).any(axis=(1, 2))
    thr = -0.01 - 1e-9
    a_abs = d_r < thr
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(big_m0 > 0, (big_m3 - big_m0) / np.maximum(big_m0, 1), 0.0)
    a_rel10 = rel < -0.10 - 1e-9
    a_rel05 = rel < -0.05 - 1e-9
    info = {
        "m0_ge25": big_m0 >= 25,
        "m0_ge50": big_m0 >= 50,
        "m0_ge25_trip3": (big_m0 >= 25) & (n_trip_matched >= 3),
        "gt10_trip3": (n_gt_matched >= 10) & (n_trip_matched >= 3),
        "none": np.ones(reps, bool),
    }
    out = {
        "mean_dR_pp": float(d_r.mean() * 100),
        "sd_dR_pp": float(d_r.std() * 100),
        "mean_R0hat": float((big_m0 / n5).mean()),
        "p_info": {k: float(v.mean()) for k, v in info.items()},
        "p_alarm": {},
    }
    for iname, iv in info.items():
        out["p_alarm"][f"abs1|{iname}"] = float((a_abs & iv).mean())
        out["p_alarm"][f"rel10|{iname}"] = float((a_rel10 & iv).mean())
        out["p_alarm"][f"rel05|{iname}"] = float((a_rel05 & iv).mean())
    out["p_alarm"]["abs1_or_rel10|m0_ge25"] = float(((a_abs | a_rel10) & info["m0_ge25"]).mean())
    out["p_alarm"]["gt_lost3|none"] = float(gt_lost3.mean())
    out["p_any_loss"] = float((big_m3 < big_m0).mean())
    return out


def coverage(design, var, r0, reps, n_boot, rng):
    """Coverage of 95 % intervals for pooled S0 recall (GT x fold) under the model."""
    tot = float(np.sqrt(sum(var)))
    tau = -stats.norm.ppf(r0) * tot
    lab = design["labeled"]
    z, a = draw_z(design, var, reps, rng)
    m0 = (z > tau) & lab[None, :, :, None]
    t_idx = design["pipe_t"]
    road_of_t = np.repeat(np.arange(4), ROAD_TRIPLETS)
    # per-triplet sums (all folds) and denominators
    hits_t = np.zeros((reps, 16))
    den_t = np.zeros(16)
    for t in range(16):
        sel = t_idx == t
        hits_t[:, t] = m0[:, sel].sum(axis=(1, 2, 3))
        den_t[t] = lab[sel].sum() * N_FOLDS
    # per-GT (pipe-line) hits for GT-level bootstrap
    gt_hits = m0.sum(axis=3)[:, lab]  # (reps, n)
    n = design["n"]
    est = hits_t.sum(axis=1) / den_t.sum()
    # truths
    resid_sd = np.sqrt(tot**2 - var[0])
    p_road = stats.norm.cdf((a - tau) / resid_sd)  # (reps, 4)
    den_r = np.array([den_t[road_of_t == r].sum() for r in range(4)])
    truth_cond = (p_road * den_r).sum(axis=1) / den_r.sum()
    res = {}

    def cov(lo, hi):
        return {
            "cover_superpop": float(((lo <= r0) & (r0 <= hi)).mean()),
            "cover_roadcond": float(((lo <= truth_cond) & (truth_cond <= hi)).mean()),
            "mean_width_pp": float((hi - lo).mean() * 100),
        }

    # 1 naive Wilson on 5n GT x fold treated independent
    k = hits_t.sum(axis=1)
    nn = den_t.sum()
    zq = 1.959963984540054
    ph = k / nn
    cen = (ph + zq**2 / (2 * nn)) / (1 + zq**2 / nn)
    hw = zq * np.sqrt(ph * (1 - ph) / nn + zq**2 / (4 * nn**2)) / (1 + zq**2 / nn)
    res["wilson_gt_fold_naive"] = cov(cen - hw, cen + hw)
    # 2 GT-level bootstrap (fold-common): resample GT
    idx = rng.integers(0, n, size=(n_boot, n))
    lo = np.empty(reps)
    hi = np.empty(reps)
    for i in range(reps):
        bs = gt_hits[i][idx].sum(axis=1) / (n * N_FOLDS)
        lo[i], hi[i] = np.quantile(bs, [0.025, 0.975])
    res["boot_gt_foldcommon"] = cov(lo, hi)
    # 3 triplet bootstrap unstratified
    idx = rng.integers(0, 16, size=(n_boot, 16))
    for i in range(reps):
        bs = hits_t[i][idx].sum(axis=1) / den_t[idx].sum(axis=1)
        lo[i], hi[i] = np.quantile(bs, [0.025, 0.975])
    res["boot_triplet_unstrat"] = cov(lo, hi)
    # 4 triplet bootstrap stratified by road (CR2 single triplet fixed)
    strat_idx = []
    for r in range(4):
        members = np.where(road_of_t == r)[0]
        strat_idx.append(members[rng.integers(0, len(members), size=(n_boot, len(members)))])
    sidx = np.concatenate(strat_idx, axis=1)
    for i in range(reps):
        bs = hits_t[i][sidx].sum(axis=1) / den_t[sidx].sum(axis=1)
        lo[i], hi[i] = np.quantile(bs, [0.025, 0.975])
    res["boot_triplet_strat_road"] = cov(lo, hi)
    # 5 road bootstrap (4 clusters)
    hits_r = np.stack([hits_t[:, road_of_t == r].sum(axis=1) for r in range(4)], axis=1)
    idx = rng.integers(0, 4, size=(n_boot, 4))
    for i in range(reps):
        bs = hits_r[i][idx].sum(axis=1) / den_r[idx].sum(axis=1)
        lo[i], hi[i] = np.quantile(bs, [0.025, 0.975])
    res["boot_road_4"] = cov(lo, hi)

    # 6 ratio estimator, cluster-robust SE with t df = G-1 (triplets, roads)
    def robust(hits, den, df):
        g = hits.shape[1]
        r_hat = hits.sum(axis=1) / den.sum()
        resid = hits - r_hat[:, None] * den[None, :]
        var_r = g / (g - 1) * (resid**2).sum(axis=1) / den.sum() ** 2
        tq = stats.t.ppf(0.975, df)
        se = np.sqrt(var_r)
        return r_hat - tq * se, r_hat + tq * se

    lo6, hi6 = robust(hits_t, den_t, 15)
    res["robust_t_triplet_df15"] = cov(lo6, hi6)
    lo7, hi7 = robust(hits_r, den_r, 3)
    res["robust_t_road_df3"] = cov(lo7, hi7)
    # 8 leave-one-road-out range (min-max), descriptive
    loro = np.stack(
        [(k - hits_r[:, r]) / (nn - den_r[r]) for r in range(4)], axis=1
    )
    res["loro_range"] = cov(loro.min(axis=1), loro.max(axis=1))
    res["mean_est"] = float(est.mean())
    return res


def main(out_json, reps, cov_reps, n_boot):
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    results = {"seed": SEED, "reps": reps, "cov_reps": cov_reps, "n_boot": n_boot,
               "road_triplets": ROAD_TRIPLETS, "q_lab": Q_LAB, "var_scenarios": VAR_SCEN,
               "sus_var": [SUS_ROAD_VAR, SUS_PIPE_VAR], "designs": {}, "alarm": [], "coverage": [],
               "selfcheck": {}}
    n_grid = [50, 100, 150, 190]
    designs = {}
    for n in n_grid:
        d = make_design(n, np.random.default_rng(SEED + n))
        designs[n] = d
        results["designs"][str(n)] = {"n": d["n"], "n_pipes": d["n_pipes"], "alloc": d["alloc"]}
    r0_grid = [0.02, 0.05, 0.10, 0.20, 0.40]
    delta_grid = [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05, -0.075, -0.10]
    band = 0.5

    # self checks
    d = designs[100]
    chk = simulate_alarm(d, VAR_SCEN["base"], 0.2, 0.0, 0.9, 0.0, 0.5, band, 2000, rng)
    results["selfcheck"]["null_zero_action_alarm"] = chk["p_alarm"]["abs1|none"]
    chk = simulate_alarm(d, VAR_SCEN["base"], 0.2, 1.0, 1.0, 0.0, 0.5, band, 2000, rng)
    results["selfcheck"]["all_lost_mean_dR_pp(expect -20)"] = chk["mean_dR_pp"]
    results["selfcheck"]["all_lost_alarm(expect ~1)"] = chk["p_alarm"]["abs1|m0_ge25"]
    chk = simulate_alarm(d, VAR_SCEN["base"], 0.2, 0.25, 0.8, 0.0, 0.5, band, 4000, rng)
    results["selfcheck"]["mean_dR_pp(expect -4.0)"] = chk["mean_dR_pp"]

    for vname, var in VAR_SCEN.items():
        if vname != "base":
            n_list, r_list = [50, 190], [0.05, 0.20]
        else:
            n_list, r_list = n_grid, r0_grid
        for n in n_list:
            design = designs[n]
            for r0 in r_list:
                tot = float(np.sqrt(sum(var)))
                tau_s = -stats.norm.ppf(r0)
                p_near = stats.norm.cdf(tau_s) - stats.norm.cdf(tau_s - band)
                # null-ish: balanced gross 1 %p loss and 1 %p gain
                cells = []
                for gross in [0.01, 0.02]:
                    h = 0.9
                    pl = gross / (r0 * h)
                    pg = gross / (p_near * 0.5)
                    if pl <= 1 and pg <= 1:
                        cells.append(("balanced_gross%g" % (gross * 100), 0.0, pl, h, pg))
                for delta in delta_grid:
                    for h in ([0.9, 0.5] if vname == "base" else [0.9]):
                        pl = -delta / (r0 * h)
                        if pl <= 1:
                            cells.append(("loss_only_h%g" % h, delta, pl, h, 0.0))
                        else:
                            results["alarm"].append({"var": vname, "n": n, "r0": r0, "kind": "loss_only_h%g" % h,
                                                     "delta_pp": delta * 100, "unreachable": True})
                for kind, delta, pl, h, pg in cells:
                    o = simulate_alarm(design, var, r0, pl, h, pg, 0.5, band, reps, rng)
                    o.update({"var": vname, "n": n, "n_actual": design["n"], "r0": r0, "kind": kind,
                              "delta_pp": delta * 100, "pi_loss": pl, "h": h, "pi_gain": pg,
                              "unreachable": False})
                    results["alarm"].append(o)
        print(f"[status] var={vname} done t={time.time() - t0:.0f}s", flush=True)

    for vname in ["base", "weak_road", "strong_road"]:
        for n in [50, 190]:
            for r0 in [0.05, 0.20]:
                c = coverage(designs[n], VAR_SCEN[vname], r0, cov_reps, n_boot, rng)
                c.update({"var": vname, "n": n, "n_actual": designs[n]["n"], "r0": r0})
                results["coverage"].append(c)
    print(f"[status] coverage done t={time.time() - t0:.0f}s", flush=True)
    results["elapsed_s"] = time.time() - t0
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps({"status": "ok", "out": out_json, "elapsed_s": round(results["elapsed_s"], 1)}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
