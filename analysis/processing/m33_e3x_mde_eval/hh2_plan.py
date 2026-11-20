"""HH2 power on the sealed plan structure (partial eta^2 of model after background, basic interval).

Rule (m33_e3x_plan.md v0.4 sections 3-2, 3-3): position value Y_on, additive background + model
fixed effects, SS_model Type II after background, eta2_p = SS_m / (SS_m + SS_res); basic interval
[2 eta - q_{1-a}, 2 eta - q_a], a = 0.05/12, stratified cluster bootstrap; confirmed iff lower > 0.5.
Truth eta2_p of each scenario = value on a large design with the plan's model / background mix.
Outcomes are synthetic (same latent-threshold model as the [STV] simulator).

Usage: hh2_plan.py <plan_dir> <out.json> <R> <B>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3x_plan_mde as S  # noqa: E402

A = 0.05 / 12


def design(models, bgs, model_list):
    mi = {m: k for k, m in enumerate(model_list)}
    X = np.zeros((len(models), 1 + len(model_list)))
    X[:, 0] = (np.asarray(bgs) == "ids").astype(float)
    for r, m in enumerate(models):
        X[r, 1 + mi[m]] = 1.0
    return X  # model dummies span the intercept


def wls_ssres(W, X, y, ridge=1e-9):
    n, p = X.shape
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    Am = (W @ XX).reshape(-1, p, p) + ridge * np.eye(p)
    rhs = W @ (X * y[:, None])
    beta = np.linalg.solve(Am, rhs[..., None])[..., 0]
    return W @ (y**2) - np.sum(beta * rhs, axis=1)


def eta(W, models, bgs, y, ml):
    X = design(models, bgs, ml)
    X0 = np.stack([np.ones(len(y)), X[:, 0]], axis=1)
    ss_full = np.maximum(wls_ssres(W, X, y), 0)
    ss_bg = np.maximum(wls_ssres(W, X0, y), 0)
    ss_m = np.maximum(ss_bg - ss_full, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ss_m / (ss_m + ss_full)


def gen(rng, cfg, models, bgs, clus):
    tau = cfg["tau_h"]
    mu = np.array([S.STAGEC_MU[m] for m in models])
    q = S.q_for(mu, tau)
    qbar = S.q_for(np.array([0.4]), tau)[0]
    q = qbar + cfg["spread"] * (q - qbar) + cfg.get("q_shift", 0.0)
    icc = {"4tu": cfg["icc_h_4tu"], "ids": cfg["icc_h_ids"]}
    a = S.position_latent(rng, clus, q, icc, bgs, tau)
    ef = rng.standard_normal((len(models), 5))
    u = S.gen_units(rng, a, 4, 0.0, cfg["s_w2"], efold=ef, rho_f=cfg["rho_f"])
    return u.mean(axis=1)


def boot_w(rng, clus, bgs, B):
    out = np.zeros((B, len(clus)))
    by = {}
    for i, (c, g) in enumerate(zip(clus, bgs)):
        by.setdefault(g, {}).setdefault(c, []).append(i)
    for g, d in by.items():
        keys = list(d)
        cnt = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=B)
        for k, c in enumerate(keys):
            out[:, d[c]] = cnt[:, [k]]
    return out


def main():
    plan_dir, outp, R, B = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    H, _ = S.load_plan(plan_dir)
    models, bgs, clus = list(H["model"]), list(H["bg"]), list(H["cluster"])
    ml = sorted(set(models))
    base = dict(tau_h=0.8, s_w2=0.1, rho_f=0.85, icc_h_4tu=0.28, icc_h_ids=0.42)
    rng = np.random.default_rng(20261124)
    # big truth design: plan mix x 150
    bm, bb, bc = [], [], []
    for k in range(150):
        bm += models
        bb += bgs
        bc += [f"{c}#{k}" for c in clus]
    t0 = time.time()
    res = {"R": R, "B": B, "a": A, "scenarios": {}}
    for sp in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8):
        cfg = base | {"spread": sp}
        ty = gen(rng, cfg, bm, bb, bc)
        truth = float(eta(np.ones((1, len(ty))), bm, bb, ty, ml)[0])
        lows, ests = [], []
        for _ in range(R):
            y = gen(rng, cfg, models, bgs, clus)
            e0 = float(eta(np.ones((1, len(y))), models, bgs, y, ml)[0])
            eb = eta(boot_w(rng, clus, bgs, B), models, bgs, y, ml)
            ql, qh = np.nanquantile(eb, [A, 1 - A])
            lows.append((e0, 2 * e0 - qh, 2 * e0 - ql))
        r = np.array(lows)
        res["scenarios"][f"{sp}"] = {
            "truth_eta2p": truth, "mean_hat": float(r[:, 0].mean()), "sd_hat": float(r[:, 0].std(ddof=1)),
            "p_confirmed": float(np.mean(r[:, 1] > 0.5)), "p_reversed": float(np.mean(r[:, 2] < 0.5)),
            "coverage": float(np.mean((r[:, 1] <= truth) & (r[:, 2] >= truth))),
            "mcse_p": float(np.sqrt(max(np.mean(r[:, 1] > 0.5) * (1 - np.mean(r[:, 1] > 0.5)), 1e-12) / R))}
    res["elapsed_s"] = round(time.time() - t0, 1)
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(outp), "elapsed_s": res["elapsed_s"]}))


if __name__ == "__main__":
    main()
