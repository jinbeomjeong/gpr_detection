"""M2-10 (a) meta eps_r residual bias - exploration on already-seen 4TU outputs.

Implements docs/processing/m210_a_eps_bias.md v1.0 sections 0-2 and 3 (settings
fixed before this script ran). Read only: it changes no 4TU rule or output; the
per-site eps values are size diagnostics, not inputs to any depth.

Inputs (committed csv): depth_4tu_v2_matches.csv, depth_4tu_v2_candidates.csv,
depth_4tu_v2_summary.csv (reference), data/processing/m210d_truth_full.csv (hash only).
Outputs (docs/processing/, LF, relative paths only):
  m210_a_eps_bias_sites.csv, m210_a_eps_bias_forms.csv, m210_a_eps_bias.json,
  m210_a_ratio.png, m210_a_scatter.png, m210_a_k.png

Run (single process, BelowNormal, E cores 24-27, MKL_CBWR=COMPATIBLE):
    conda run -n gpr-dev python -I analysis/processing/m210a_eps_bias.py
"""

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "processing"
SPEC = DOC / "m210_a_eps_bias.md"
INPUTS = {
    "matches": DOC / "depth_4tu_v2_matches.csv",
    "candidates": DOC / "depth_4tu_v2_candidates.csv",
    "summary": DOC / "depth_4tu_v2_summary.csv",
    "truth_full": ROOT / "data" / "processing" / "m210d_truth_full.csv",
}
IDS = {"activity_id": str, "path": str, "line_id": str}
SEED = 20261121
N_BOOT = 2000
N_PERM = 10000
EPS_LO, EPS_HI, A_MAX = 3.0, 36.0, 0.13
SENS = [(lo, hi, a) for lo in (3.0, 4.0) for hi in (30.0, 36.0) for a in (0.075, 0.13)]
F_SCALE = 0.10
MIN_PAIRS, MIN_ACTS = 10, 4
HEADER_SITES = (1, 4, 12, 13)
MEASURED_SITES = (3, 5, 6, 8, 11)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def admissible(r, d, eps_meta, lo=EPS_LO, hi=EPS_HI, a=A_MAX):
    low = np.sqrt(lo / eps_meta) - a / d
    high = np.sqrt(hi / eps_meta) + a / d
    return (r >= low) & (r <= high)


def huber_fit(x, y, model):
    """Huber (least_squares, f_scale 0.10 m) fit of y on x for model k / a / ka."""
    if len(x) == 0:
        return None
    if model == "k":
        fun, p0 = (lambda p: y - p[0] * x), [1.0]
    elif model == "a":
        fun, p0 = (lambda p: y - x - p[0]), [0.0]
    else:
        fun, p0 = (lambda p: y - p[0] * x - p[1]), [1.0, 0.0]
    if model == "ka" and len(x) < 3:
        return None
    res = least_squares(fun, p0, loss="huber", f_scale=F_SCALE)
    return res.x


def predict(params, model, x):
    if model == "k":
        return params[0] * x
    if model == "a":
        return x + params[0]
    return params[0] * x + params[1]


def metrics(e):
    e = np.asarray(e, dtype=float)
    if len(e) == 0:
        return {"bias": None, "mae": None, "within_030": None}
    return {
        "bias": float(np.mean(e)),
        "mae": float(np.mean(np.abs(e))),
        "within_030": float(np.mean(np.abs(e) <= 0.30 + 1e-9)),
    }


def rnd(x, n=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def load():
    m = pd.read_csv(INPUTS["matches"], dtype=IDS)
    c = pd.read_csv(INPUTS["candidates"], dtype=IDS)
    eps_act = c.groupby("activity_id")["eps_r_used"].agg(lambda s: float(s.dropna().unique()[0]))
    m = m[(m.gt_set == "exclude_x") & (m.zero_variant == "detected") & m.variant.isin(["main", "b"])].copy()
    m["eps_meta"] = m["activity_id"].map(eps_act)
    m["d"] = m["gt_depth_m"].astype(float)
    m["dhat"] = m["est_depth_m"].astype(float)
    m["r"] = m["dhat"] / m["d"]
    m["e"] = m["dhat"] - m["d"]
    m["group"] = m["site"].astype(int).astype(str) + "|eps" + m["eps_meta"].map(lambda v: f"{v:g}")
    return m, c, eps_act


def group_analysis(g: pd.DataFrame, rng: np.random.Generator, do_boot: bool):
    eps_meta = float(g["eps_meta"].iloc[0])
    adm = admissible(g["r"].values, g["d"].values, eps_meta)
    ga = g[adm]
    acts = sorted(g["activity_id"].unique())
    acts_adm = sorted(ga["activity_id"].unique())
    out = {
        "eps_meta": eps_meta,
        "n_pairs": int(len(g)),
        "n_activities": len(acts),
        "n_admissible": int(adm.sum()),
        "n_activities_admissible": len(acts_adm),
        "n_below_window": int((~adm & (g["r"].values < 1)).sum()),
        "n_above_window": int((~adm & (g["r"].values >= 1)).sum()),
    }
    out["frac_outside"] = rnd(1 - out["n_admissible"] / out["n_pairs"]) if out["n_pairs"] else None
    out["small_sample"] = out["n_admissible"] < MIN_PAIRS or len(acts_adm) < MIN_ACTS
    if len(ga) == 0:
        return out, None
    k_med = float(np.median(ga["r"]))
    k_hub = float(huber_fit(ga["d"].values, ga["dhat"].values, "k")[0])
    out.update(
        k_median=rnd(k_med), k_huber=rnd(k_hub),
        eps_true_huber=rnd(eps_meta * k_hub**2, 3), d_eps_huber=rnd(eps_meta * (k_hub**2 - 1), 3),
    )
    before = metrics(g["e"].values)
    after = metrics(g["dhat"].values / k_hub - g["d"].values)
    out.update({f"{k}_before": rnd(v) for k, v in before.items()})
    out.update({f"{k}_after_free": rnd(v) for k, v in after.items()})
    out["share_S_free"] = rnd(1 - abs(after["bias"]) / abs(before["bias"])) if before["bias"] else None
    # leave-one-activity-out
    loo_e, n_loo = [], 0
    for a in acts:
        other = ga[ga.activity_id != a]
        if len(other) < 3:
            continue
        k_o = huber_fit(other["d"].values, other["dhat"].values, "k")[0]
        ga_ = g[g.activity_id == a]
        loo_e.extend((ga_["dhat"].values / k_o - ga_["d"].values).tolist())
        n_loo += 1
    loo = metrics(loo_e)
    out.update({f"{k}_after_loo": rnd(v) for k, v in loo.items()})
    out["n_activities_loo"] = n_loo
    out["share_S_loo"] = (
        rnd(1 - abs(loo["bias"]) / abs(before["bias"])) if before["bias"] and loo["bias"] is not None else None
    )
    if do_boot and len(acts_adm) >= 2:
        by_act = {a: ga[ga.activity_id == a] for a in acts_adm}
        ks = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(acts_adm, size=len(acts_adm), replace=True)
            s = pd.concat([by_act[a] for a in pick])
            ks[b] = huber_fit(s["d"].values, s["dhat"].values, "k")[0]
        out["k_huber_ci95"] = [rnd(np.quantile(ks, 0.025)), rnd(np.quantile(ks, 0.975))]
        out["p_k_ne_1"] = rnd(min(1.0, 2 * min(np.mean(ks <= 1), np.mean(ks >= 1))))
    return out, ga


def form_analysis(ga: pd.DataFrame, rng: np.random.Generator):
    acts = sorted(ga["activity_id"].unique())
    out = {"n_admissible": int(len(ga)), "n_activities": len(acts)}
    for model in ("k", "a", "ka"):
        p = huber_fit(ga["d"].values, ga["dhat"].values, model)
        out[f"M_{model}_params"] = None if p is None else [rnd(v) for v in p]
    loo = {model: {} for model in ("k", "a", "ka")}
    for a in acts:
        other, held = ga[ga.activity_id != a], ga[ga.activity_id == a]
        if len(other) < 3:
            continue
        for model in ("k", "a", "ka"):
            p = huber_fit(other["d"].values, other["dhat"].values, model)
            if p is None:
                continue
            loo[model][a] = np.abs(held["dhat"].values - predict(p, model, held["d"].values))
    for model in loo:
        vals = np.concatenate(list(loo[model].values())) if loo[model] else np.array([])
        out[f"M_{model}_loo_median_abs"] = rnd(np.median(vals)) if len(vals) else None
    common = sorted(set(loo["k"]) & set(loo["a"]))
    if len(common) >= 2:
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(common, size=len(common), replace=True)
            ek = np.concatenate([loo["k"][a] for a in pick])
            ea = np.concatenate([loo["a"][a] for a in pick])
            diffs[b] = np.median(ek) - np.median(ea)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        out["Mk_minus_Ma_ci95"] = [rnd(lo), rnd(hi)]
        out["form"] = "multiplicative" if hi < 0 else "additive" if lo > 0 else "not_separable"
    else:
        out["Mk_minus_Ma_ci95"], out["form"] = None, "insufficient"
    # r vs 1/d (slope = a, intercept = k)
    x, y = 1 / ga["d"].values, ga["r"].values
    if len(ga) >= 3:
        slope, icpt = np.polyfit(x, y, 1)
        out["r_on_inv_d"] = {"intercept_k": rnd(icpt), "slope_a_m": rnd(slope)}
        if len(acts) >= 2:
            by_act = {a: ga[ga.activity_id == a] for a in acts}
            bs = np.empty((N_BOOT, 2))
            for b in range(N_BOOT):
                s = pd.concat([by_act[a] for a in rng.choice(acts, size=len(acts), replace=True)])
                xs, ys = 1 / s["d"].values, s["r"].values
                bs[b] = np.polyfit(xs, ys, 1) if len(np.unique(xs)) > 1 else (np.nan, np.nan)
            out["r_on_inv_d"]["slope_ci95"] = [rnd(np.nanquantile(bs[:, 0], 0.025)), rnd(np.nanquantile(bs[:, 0], 0.975))]
            out["r_on_inv_d"]["intercept_ci95"] = [rnd(np.nanquantile(bs[:, 1], 0.025)), rnd(np.nanquantile(bs[:, 1], 0.975))]
    return out


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = rnd(running)
    return out


def main() -> None:
    m, c, eps_act = load()
    rng = np.random.default_rng(SEED)
    site_rows, form_rows, admissible_sets = [], [], {}
    for variant in ("main", "b"):
        for assumption in ("top", "center"):
            sub = m[(m.variant == variant) & (m.assumption == assumption)]
            pvals = {}
            for group, g in sorted(sub.groupby("group"), key=lambda kv: (int(kv[0].split("|")[0]), kv[0])):
                res, ga = group_analysis(g, rng, do_boot=True)
                row = {"variant": variant, "assumption": assumption, "group": group,
                       "site": int(group.split("|")[0]), **res}
                site_rows.append(row)
                pvals[group] = res.get("p_k_ne_1")
                admissible_sets[(variant, assumption, group)] = ga
                if ga is not None and len(ga) >= 3:
                    f = form_analysis(ga, rng)
                    form_rows.append({"variant": variant, "assumption": assumption, "group": group,
                                      "small_sample": res["small_sample"], **f})
            hp = holm({k: v for k, v in pvals.items()})
            for row in site_rows:
                if row["variant"] == variant and row["assumption"] == assumption:
                    row["p_holm"] = hp.get(row["group"])
    # sensitivity (main, top)
    sens = []
    sub = m[(m.variant == "main") & (m.assumption == "top")]
    for lo, hi, a in SENS:
        for group, g in sub.groupby("group"):
            ok = admissible(g["r"].values, g["d"].values, float(g["eps_meta"].iloc[0]), lo, hi, a)
            ga = g[ok]
            k = huber_fit(ga["d"].values, ga["dhat"].values, "k")[0] if len(ga) else None
            sens.append({"eps_lo": lo, "eps_hi": hi, "a_max": a, "group": group, "n_pairs": int(len(g)),
                         "n_admissible": int(ok.sum()), "k_huber": rnd(k)})
    sub_all = sub.copy()
    tot = {}
    for lo, hi, a in SENS:
        ok = np.zeros(len(sub_all), bool)
        for i, (r, d, e) in enumerate(zip(sub_all["r"], sub_all["d"], sub_all["eps_meta"])):
            ok[i] = admissible(np.array([r]), np.array([d]), e, lo, hi, a)[0]
        tot[f"lo{lo:g}_hi{hi:g}_a{a:g}"] = {"n_pairs": int(len(sub_all)), "n_outside": int((~ok).sum())}
    # header default vs measured (main, top, admissible)
    ga_all = pd.concat([v for (vv, aa, g), v in admissible_sets.items() if vv == "main" and aa == "top" and v is not None])
    hm = ga_all[ga_all.site.isin(HEADER_SITES + MEASURED_SITES)].copy()
    hm["grp"] = np.where(hm.site.isin(HEADER_SITES), "H", "M")
    act_grp = hm.groupby("activity_id")["grp"].first()
    stat = float(np.median(hm[hm.grp == "M"]["r"]) - np.median(hm[hm.grp == "H"]["r"]))
    prng = np.random.default_rng(SEED + 1)
    acts = act_grp.index.values
    labels = act_grp.values
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        lab = dict(zip(acts, prng.permutation(labels)))
        g2 = hm["activity_id"].map(lab)
        perm[i] = np.median(hm[g2 == "M"]["r"]) - np.median(hm[g2 == "H"]["r"])
    header = {
        "statistic_median_r_M_minus_H": rnd(stat),
        "median_r_H": rnd(np.median(hm[hm.grp == "H"]["r"])),
        "median_r_M": rnd(np.median(hm[hm.grp == "M"]["r"])),
        "n_pairs_H": int((hm.grp == "H").sum()), "n_pairs_M": int((hm.grp == "M").sum()),
        "n_acts_H": int((act_grp == "H").sum()), "n_acts_M": int((act_grp == "M").sum()),
        "p_two_sided": rnd(float(np.mean(np.abs(perm) >= abs(stat) - 1e-12))),
    }
    s10 = ga_all[ga_all.site == 10]
    header["site10"] = {
        "eps9_median_r": rnd(np.median(s10[s10.eps_meta == 9.0]["r"])) if (s10.eps_meta == 9.0).any() else None,
        "eps9_n": int((s10.eps_meta == 9.0).sum()),
        "other_median_r": rnd(np.median(s10[s10.eps_meta != 9.0]["r"])) if (s10.eps_meta != 9.0).any() else None,
        "other_n": int((s10.eps_meta != 9.0).sum()),
    }
    # curvature cross-check (E3): group-level Spearman between k_huber and median qc_r
    cc = c[(c.status == "ok") & (c.qc_verdict == "in") & c.qc_r.notna()].copy()
    cc["group"] = cc["site"].astype(int).astype(str) + "|eps" + cc["eps_r_used"].map(lambda v: f"{v:g}")
    rho = cc.groupby("group")["qc_r"].agg(["median", "size"])
    main_top = [r for r in site_rows if r["variant"] == "main" and r["assumption"] == "top"]
    pairs = [(r["group"], r["k_huber"], float(rho.loc[r["group"], "median"]), int(rho.loc[r["group"], "size"]))
             for r in main_top if not r["small_sample"] and r.get("k_huber") is not None and r["group"] in rho.index]
    curv = {"groups": [{"group": g, "k_huber": k, "median_qc_r": rnd(q), "n_qc_in": n} for g, k, q, n in pairs]}
    if len(pairs) >= 4:
        kk = np.array([p[1] for p in pairs])
        qq = np.array([p[2] for p in pairs])
        rs = float(spearmanr(kk, qq).statistic)
        crng = np.random.default_rng(SEED + 2)
        perm_r = np.array([spearmanr(kk, crng.permutation(qq)).statistic for _ in range(N_PERM)])
        curv.update(spearman=rnd(rs), p_two_sided=rnd(float(np.mean(np.abs(perm_r) >= abs(rs) - 1e-12))),
                    n_groups=len(pairs), expectation="negative under an eps error (rho ~ 1/k)")
    # pooled summary (main, top): all pairs, free / LOO per-group correction
    pooled = {}
    for variant in ("main", "b"):
        for assumption in ("top", "center"):
            rows = [r for r in site_rows if r["variant"] == variant and r["assumption"] == assumption]
            n = sum(r["n_pairs"] for r in rows)
            sub = m[(m.variant == variant) & (m.assumption == assumption)]
            kmap = {r["group"]: r.get("k_huber") for r in rows}
            e0 = sub["e"].values
            k_arr = sub["group"].map(kmap).astype(float).fillna(1.0).values
            e1 = sub["dhat"].values / k_arr - sub["d"].values
            adm_n = sum(r["n_admissible"] for r in rows)
            pooled[f"{variant}|{assumption}"] = {
                "n_pairs": n, "n_admissible": adm_n, "frac_outside": rnd(1 - adm_n / n),
                "before": {k: rnd(v) for k, v in metrics(e0).items()},
                "after_free_per_group": {k: rnd(v) for k, v in metrics(e1).items()},
            }
    # write
    cols = ["variant", "assumption", "group", "site", "eps_meta", "n_pairs", "n_activities", "n_admissible",
            "n_activities_admissible", "n_below_window", "n_above_window", "frac_outside", "small_sample",
            "k_median", "k_huber", "k_huber_ci95", "p_k_ne_1", "p_holm", "eps_true_huber", "d_eps_huber",
            "bias_before", "mae_before", "within_030_before", "bias_after_free", "mae_after_free",
            "within_030_after_free", "share_S_free", "n_activities_loo", "bias_after_loo", "mae_after_loo",
            "within_030_after_loo", "share_S_loo"]
    df = pd.DataFrame(site_rows)
    for col in cols:
        if col not in df:
            df[col] = None
    df["k_huber_ci95"] = df["k_huber_ci95"].map(lambda v: "" if not isinstance(v, list) else f"{v[0]};{v[1]}")
    df[cols].to_csv(DOC / "m210_a_eps_bias_sites.csv", index=False, lineterminator="\n")
    fdf = pd.DataFrame(form_rows)
    for col in ("M_k_params", "M_a_params", "M_ka_params", "Mk_minus_Ma_ci95"):
        fdf[col] = fdf[col].map(lambda v: "" if not isinstance(v, list) else ";".join(str(x) for x in v))
    fdf["r_on_inv_d"] = fdf["r_on_inv_d"].map(lambda v: "" if not isinstance(v, dict) else json.dumps(v))
    fdf.to_csv(DOC / "m210_a_eps_bias_forms.csv", index=False, lineterminator="\n")
    summary = {
        "note": "M2-10 (a) exploration on already-seen 4TU outputs; not applied to any 4TU rule",
        "spec": "docs/processing/m210_a_eps_bias.md v1.0",
        "spec_sha256_at_run": sha256(SPEC),
        "inputs_sha256": {k: sha256(p) for k, p in INPUTS.items()},
        "inputs": {k: str(p.relative_to(ROOT)).replace("\\", "/") for k, p in INPUTS.items()},
        "settings": {"eps_lo": EPS_LO, "eps_hi": EPS_HI, "a_max": A_MAX, "f_scale": F_SCALE, "seed": SEED,
                     "n_boot": N_BOOT, "n_perm": N_PERM, "min_pairs": MIN_PAIRS, "min_acts": MIN_ACTS},
        "pooled": pooled,
        "sensitivity_main_top_totals": tot,
        "sensitivity_main_top_groups": sens,
        "header_default_vs_measured": header,
        "curvature_crosscheck": curv,
    }
    (DOC / "m210_a_eps_bias.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                              encoding="utf-8", newline="\n")
    figures(m, df, curv)
    print(json.dumps({"ok": True, "groups": len(site_rows), "forms": len(form_rows)}))


def figures(m, df, curv):
    sub = m[(m.variant == "main") & (m.assumption == "top")]
    groups = sorted(sub["group"].unique(), key=lambda g: (int(g.split("|")[0]), g))
    # 1 ratio strip with eps window
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, g in enumerate(groups):
        gg = sub[sub.group == g]
        eps = float(gg["eps_meta"].iloc[0])
        ok = admissible(gg["r"].values, gg["d"].values, eps)
        jitter = (np.random.default_rng(SEED + 10 + i).random(len(gg)) - 0.5) * 0.3
        ax.scatter(i + jitter[ok], gg["r"].values[ok], s=14, color="#1f5f99", label="허용 범위 안" if i == 0 else None)
        ax.scatter(i + jitter[~ok], gg["r"].values[~ok], s=14, color="#c0392b", marker="x",
                   label="εr로 설명 불가(범위 밖)" if i == 0 else None)
        ax.plot([i - 0.35, i + 0.35], [math.sqrt(EPS_LO / eps)] * 2, color="#555", lw=1)
        ax.plot([i - 0.35, i + 0.35], [math.sqrt(EPS_HI / eps)] * 2, color="#555", lw=1, ls="--")
    ax.axhline(1, color="#999", lw=0.8)
    ax.set_xticks(range(len(groups)), [g.replace("|eps", "\nεr ") for g in groups], fontsize=8)
    ax.set_ylabel("깊이 비 r = 추정/정답")
    ax.set_title("현장별 추정/정답 깊이 비: 실선·점선 = 실제 εr 3·36에서 나올 수 있는 비(가산 여유 전), 상단·검출·주 짝")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOC / "m210_a_ratio.png", dpi=130)
    plt.close(fig)
    # 2 scatter small multiples
    n = len(groups)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
    for ax, g in zip(axes.flat, groups):
        gg = sub[sub.group == g]
        ax.scatter(gg["d"], gg["dhat"], s=10, color="#1f5f99")
        lim = [0, max(2.5, float(gg[["d", "dhat"]].max().max()) * 1.05)]
        ax.plot(lim, lim, color="#999", lw=0.8)
        row = df[(df.variant == "main") & (df.assumption == "top") & (df.group == g)]
        if len(row) and pd.notna(row["k_huber"].iloc[0]):
            k = float(row["k_huber"].iloc[0])
            ax.plot(lim, [k * lim[0], k * lim[1]], color="#d98c1a", lw=1.2, label=f"M_k k={k:.2f}")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(g.replace("|eps", " εr "), fontsize=9)
        ax.set_xlabel("정답 깊이 (m)", fontsize=8)
        ax.set_ylabel("추정 깊이 (m)", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("현장별 추정 대 정답 깊이 (상단·검출·주 짝), 회색 = 1:1, 주황 = 상수배 모형", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOC / "m210_a_scatter.png", dpi=120)
    plt.close(fig)
    # 3 k with CI and curvature
    d = df[(df.variant == "main") & (df.assumption == "top") & df["k_huber"].notna()].copy()
    d = d.sort_values(["site", "group"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (_, r) in enumerate(d.iterrows()):
        color = "#c0392b" if r["site"] in HEADER_SITES else "#1f5f99"
        ci = r["k_huber_ci95"]
        if isinstance(ci, str) and ";" in ci:
            lo, hi = (float(x) for x in ci.split(";"))
            ax.plot([i, i], [lo, hi], color=color, lw=2, alpha=0.6)
        ax.scatter(i, r["k_huber"], color=color, s=30, marker="o" if not r["small_sample"] else "s",
                   facecolors=color if not r["small_sample"] else "none")
    curv_map = {x["group"]: x["median_qc_r"] for x in curv.get("groups", [])}
    xs = [i for i, g in enumerate(d["group"]) if g in curv_map]
    if xs:
        ax.scatter(xs, [1 / curv_map[g] for g in d["group"] if g in curv_map], marker="^", color="#2e8b57",
                   s=28, label="1/중앙 곡률 비(참고, 치우침 있음)")
    ax.axhline(1, color="#999", lw=0.8)
    ax.set_xticks(range(len(d)), [g.replace("|eps", "\nεr ") for g in d["group"]], fontsize=8)
    ax.set_ylabel("현장 배율 k (Huber, 95 % 구간)")
    ax.set_title("현장 배율 k: 빨강 = 헤더 기본값 εr 9 현장, 파랑 = 그 밖, 빈 네모 = 표본 부족")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOC / "m210_a_k.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
