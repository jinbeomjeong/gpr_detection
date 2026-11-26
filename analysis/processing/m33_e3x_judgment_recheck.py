"""Independent re-aggregation of the M3-3 E3x judgment from record csv files.

Written from the rule text of docs/processing/m33_e3x_plan.md v0.5 (sec. 3-2 to
3-4), without importing analysis/detection/m33_e3x_judge.py.  Reads only the
retry outputs runs/eval/m33/e3x_retry/{injected,judgment}/ and the sealed plan
tables runs/eval/m33/e3x/plan/.  Never touches the first-try folders.

Output: analysis/processing/m33_e3x_judgment_recheck.json (point estimates,
counts, floor values, M* sets, eta2_p, and an own cluster bootstrap with the
Satterthwaite interval for robustness of the decisions).

Bootstrap: own implementation, seed 20261121 (numpy default_rng), B = 20000,
per hypothesis per background (4tu then gz_ids) an integer draw matrix
rng.integers(0, G, (B, G)) over sorted cluster ids.  The draw order of the
sealed tool is not reproduced on purpose; agreement is judged on decisions and
interval ends to about 1e-2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RETRY = ROOT / "runs" / "eval" / "m33" / "e3x_retry"
INJ = RETRY / "injected"
JDG = RETRY / "judgment"
OUT = ROOT / "analysis" / "processing" / "m33_e3x_judgment_recheck.json"

FIRST_TRY = ROOT / "runs" / "eval" / "m33" / "e3x"
for p in (INJ, JDG):
    assert not p.resolve().is_relative_to((FIRST_TRY / "injected").resolve())

CI_LEVEL = 1 - 0.05 / 6
TAIL = 0.05 / 12
DELTA = 0.05
FLOOR = 0.10
B = 20000
SEED = 20261121
SJ = (12.0, 18.0)
BGS = ("4tu", "gz_ids")


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1"])


# ---------------------------------------------------------------- hyperbola
h = pd.read_csv(INJ / "hyperbola_records.csv", dtype={"cluster_id": str, "line_id": str}, low_memory=False)
h["accepted_b"] = truthy(h["accepted"])
ha = h[h["accepted_b"]].copy()
# unit = (position, taper, snr, polarity); y_u = mean over folds 0-4
folds_per_unit = ha.groupby(["position_id", "taper", "snr_db", "polarity"])["fold"].nunique()
assert (folds_per_unit == 5).all(), "every accepted unit must have 5 folds"
hu = (
    ha.groupby(["position_id", "taper", "snr_db", "polarity"], as_index=False)
    .agg(y=("hit", "mean"))
)
pos_info = (
    h.groupby("position_id")
    .agg(
        line_id=("line_id", "first"),
        dataset=("dataset", "first"),
        cluster=("cluster_id", "first"),
        model=("model", "first"),
        shares=("shares_r3_activity", "first"),
        r_edge=("r_edge_db", "first"),
        stratum=("edge_stratum", "first"),
        snrset=("edge_snr_set", "first"),
    )
)
pos_info["cluster"] = pos_info["cluster"].astype(str)


def unit(pid, taper, snr, pol):
    key = (pid, taper, snr, pol)
    return U.get(key)


U = {(r.position_id, r.taper, float(r.snr_db), int(r.polarity)): r.y for r in hu.itertuples()}


def arm_mean(pid, taper, snrs):
    vals = [U.get((pid, taper, float(s), p)) for s in snrs for p in (1, -1)]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


rows = []
for pid, inf in pos_info.iterrows():
    y_on = arm_mean(pid, "on", SJ)
    y_off = arm_mean(pid, "off", SJ)
    rows.append(dict(position_id=pid, Y_on=y_on, Y_off=y_off))
hp = pd.DataFrame(rows).set_index("position_id").join(pos_info)

both = hp.dropna(subset=["Y_on", "Y_off"])
model_mean = both.assign(m=(both.Y_on + both.Y_off) / 2).groupby("model")["m"].mean()
m_star = sorted(model_mean[model_mean >= 0.10].index)


def edge_set(pid):
    s = pos_info.loc[pid, "snrset"]
    if isinstance(s, float) and np.isnan(s):
        return []
    return [float(x) for x in str(s).split(",") if x.strip()]


# HH1 units
hh1 = []
for pid, r in both.iterrows():
    if r.stratum != "manipulated" or r.model not in m_star:
        continue
    ss = edge_set(pid)
    assert ss, pid
    on = arm_mean(pid, "on", ss)
    off = arm_mean(pid, "off", ss)
    if on is None or off is None:
        continue
    hh1.append(dict(position_id=pid, background=r.dataset, cluster=r.cluster, model=r.model,
                    difference=on - off, floor_value=(on + off) / 2))
hh1 = pd.DataFrame(hh1)
neg = []
for pid, r in both.iterrows():
    if r.stratum != "negative_control" or r.model not in m_star:
        continue
    neg.append(dict(position_id=pid, background=r.dataset, cluster=r.cluster, model=r.model,
                    difference=r.Y_on - r.Y_off, floor_value=(r.Y_on + r.Y_off) / 2))
neg = pd.DataFrame(neg)
# HH3
hh3 = []
for pid, r in hp.iterrows():
    if r.model not in m_star:
        continue
    a = arm_mean(pid, "on", (12.0,))
    b = arm_mean(pid, "on", (30.0,))
    if a is None or b is None:
        continue
    hh3.append(dict(position_id=pid, background=r.dataset, cluster=r.cluster, model=r.model,
                    difference=a - b, floor_value=(a + b) / 2))
hh3 = pd.DataFrame(hh3)
# 30 dB rejection share among M* positions (unit-level, both tapers? -> on arm)
mstar_pos = hp[hp.model.isin(m_star)].index
rej30 = h[(h.position_id.isin(mstar_pos)) & (h.snr_db == 30) & (h.taper == "on") & (h.fold == 0)]
rej30_share_units = float((~truthy(rej30["accepted"])).mean())
pos30_missing = int(len(mstar_pos) - len(hh3))
# HH2
hh2 = hp.dropna(subset=["Y_on"]).copy()
cnt = hh2.groupby("model").size()
hh2 = hh2[hh2.model.isin(cnt[cnt >= 4].index)]

# off-arm int16 rejections among HH1 target positions
off_rej = h[(h.taper == "off") & (h.fold == 0) & (~truthy(h["accepted"]))]


# ---------------------------------------------------------------- band
b = pd.read_csv(INJ / "band_records.csv", dtype={"cluster_id": str, "line_id": str}, low_memory=False)
b["accepted_b"] = truthy(b["accepted"])
ba = b[b["accepted_b"] & (b.recipe == "r0") & (b.texture == "none")].copy()
fpu = ba.groupby(["position_id", "bin_target_px", "length_arm", "snr_db", "polarity"])["fold"].nunique()
assert (fpu == 5).all()
bu = ba.groupby(["position_id", "bin_target_px", "length_arm", "snr_db", "polarity"], as_index=False).agg(
    y=("coverage", "mean"))
BU = {(r.position_id, float(r.bin_target_px), r.length_arm, float(r.snr_db), int(r.polarity)): r.y
      for r in bu.itertuples()}
binfo = b.groupby("position_id").agg(dataset=("dataset", "first"), cluster=("cluster_id", "first"),
                                     model=("model", "first"), shares=("shares_r3_activity", "first"))
binfo["cluster"] = binfo["cluster"].astype(str)


def barm(pid, row, length, snrs=SJ, src=None):
    src = BU if src is None else src
    vals = [src.get((pid, float(row), length, float(s), p)) for s in snrs for p in (1, -1)]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def band_units(kind, src=None, snrs=SJ):
    out = []
    for pid, r in binfo.iterrows():
        y28 = barm(pid, 28, "long", snrs, src)
        y47 = barm(pid, 47, "long", snrs, src)
        y68 = barm(pid, 68, "long", snrs, src)
        y90 = barm(pid, 90, "long", snrs, src)
        s47 = barm(pid, 47, "short", snrs, src)
        s68 = barm(pid, 68, "short", snrs, src)
        if kind == "HB1":
            if None in (y28, y47, y68):
                continue
            a, bb = (y47 + y68) / 2, y28
        elif kind == "HB1b":
            if None in (y90, y47, y68):
                continue
            a, bb = (y47 + y68) / 2, y90
        elif kind == "HB2":
            if None in (y47, y68, s47, s68):
                continue
            a, bb = (y47 + y68) / 2, (s47 + s68) / 2
        out.append(dict(position_id=pid, background=r.dataset, cluster=r.cluster, model=r.model,
                        difference=a - bb, floor_value=(a + bb) / 2))
    return pd.DataFrame(out)


hb = {k: band_units(k) for k in ("HB1", "HB1b", "HB2")}


# ---------------------------------------------------------------- estimators
def est_model(df, w=None):
    w = np.ones(len(df)) if w is None else w
    vals = []
    for m, idx in df.groupby("model").indices.items():
        ww = w[idx]
        if ww.sum() == 0:
            continue
        vals.append(np.sum(ww * df.difference.values[idx]) / ww.sum())
    return float(np.mean(vals))


def est_bg(df, w=None, col="difference"):
    w = np.ones(len(df)) if w is None else w
    vals = []
    for g in BGS:
        idx = np.flatnonzero(df.background.values == g)
        if len(idx) == 0 or w[idx].sum() == 0:
            continue
        vals.append(np.sum(w[idx] * df[col].values[idx]) / w[idx].sum())
    return float(np.mean(vals))


def floor_of(df, weighting):
    d = df.assign(difference=df.floor_value)
    return est_model(d) if weighting == "model" else est_bg(d)


def est_matrix(df, W, weighting):
    """Vectorised estimator for a (B, n) weight matrix."""
    d = df.difference.values
    if weighting == "model":
        groups = df.model.values
    else:
        groups = df.background.values
    keys = sorted(set(groups))
    ind = np.stack([(groups == k).astype(float) for k in keys], axis=1)  # n x K
    num = W @ (ind * d[:, None])
    den = W @ ind
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(den > 0, num / den, np.nan)
    return np.nanmean(q, axis=1)


def counts_matrix(draws, G):
    out = np.zeros((draws.shape[0], G))
    rows = np.repeat(np.arange(draws.shape[0]), draws.shape[1])
    np.add.at(out, (rows, draws.ravel()), 1.0)
    return out


def satterthwaite(df, weighting, rng):
    est = est_model if weighting == "model" else est_bg
    point = est(df)
    clusters = {g: sorted(df.cluster[df.background == g].unique()) for g in BGS}
    v, nu_g, geff, gg = {}, {}, {}, {}
    for g in BGS:
        cl = clusters[g]
        G = len(cl)
        gg[g] = G
        draws = rng.integers(0, G, size=(B, G))
        if G < 2:
            continue
        pos_cl = df.cluster.values
        is_g = df.background.values == g
        cl_index = {c: i for i, c in enumerate(cl)}
        ci = np.array([cl_index.get(c, -1) if ig else -1 for c, ig in zip(pos_cl, is_g)])
        cnt = counts_matrix(draws, G)
        W = np.ones((B, len(df)))
        W[:, is_g] = cnt[:, ci[is_g]]
        reps = est_matrix(df, W, weighting)
        n_c = df[is_g].groupby("cluster").size().values
        geff[g] = float(n_c.sum() ** 2 / np.sum(n_c ** 2))
        nu_g[g] = max(geff[g] - 1, 1.0)
        v[g] = G / (G - 1) * float(np.var(reps, ddof=1))
    vs = sum(v.values())
    den = sum(v[g] ** 2 / nu_g[g] for g in v if v[g] > 0)
    nu = vs ** 2 / den if den > 0 else float("nan")
    half = stats.t.ppf(1 - TAIL, nu) * np.sqrt(vs)
    return dict(estimate=point, ci_low=point - half, ci_high=point + half, nu=nu,
                v_by_background=v, g_eff_by_background=geff, g_by_background=gg,
                n_positions=int(len(df)), n_clusters=int(df.cluster.nunique()))


def decide(r, F, two_sided=False):
    lo, hi = r["ci_low"], r["ci_high"]
    if lo > 0:
        return "confirmed_positive" if two_sided else "confirmed"
    if hi < 0:
        return "confirmed_negative" if two_sided else "reversed"
    if lo > -DELTA and hi < DELTA:
        return "negligible" if F >= FLOOR else "inconclusive(floor)"
    return "inconclusive"


# HH2 eta2_p (background fixed + model fixed additive, Type II for model)
_DUMMY_CACHE: dict = {}


def eta2(df, w=None):
    w = np.ones(len(df)) if w is None else w
    keep = w > 0
    y = df.Y_on.values[keep]
    sw = np.sqrt(w[keep])
    key = id(df)
    if key not in _DUMMY_CACHE:
        _DUMMY_CACHE[key] = (
            pd.get_dummies(pd.Series(df.dataset.values), drop_first=False).values.astype(float),
            pd.get_dummies(pd.Series(df.model.values), drop_first=False).values.astype(float),
        )
    Xb_full, Xm_full = _DUMMY_CACHE[key]
    X1 = Xb_full[keep]
    # full model dummies; rank handles the collinearity with the background block
    X2 = np.hstack([Xb_full, Xm_full])[keep]

    def rss(X):
        Xw = X * sw[:, None]
        yw = y * sw
        beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        r = yw - Xw @ beta
        return float(r @ r), int(rank)

    r1, k1 = rss(X1)
    r2, k2 = rss(X2)
    ss_model = r1 - r2
    df_model = k2 - k1
    n = float(w[keep].sum())
    ms_model = ss_model / df_model
    ms_res = r2 / (n - k2)
    e = ss_model / (ss_model + r2)
    om = df_model * (ms_model - ms_res) / (df_model * ms_model + (n - df_model) * ms_res)
    return e, om


def hh2_boot(df, rng, models=None):
    d = df if models is None else df[df.model.isin(models)]
    point, om_point = eta2(d)
    clusters = {g: sorted(d.cluster[d.dataset == g].unique()) for g in BGS}
    draws = {g: rng.integers(0, len(clusters[g]), size=(B, len(clusters[g]))) for g in BGS}
    reps = np.empty(B)
    cl_idx = {g: {c: i for i, c in enumerate(clusters[g])} for g in BGS}
    for bi in range(B):
        w = np.zeros(len(d))
        for g in BGS:
            counts = np.bincount(draws[g][bi], minlength=len(clusters[g]))
            m = d.dataset.values == g
            w[m] = [counts[cl_idx[g][c]] for c in d.cluster.values[m]]
        reps[bi], _ = eta2(d, w)
    qlo, qhi = np.quantile(reps, [TAIL, 1 - TAIL])
    return dict(estimate=point, omega2=om_point, ci_low=2 * point - qhi, ci_high=2 * point - qlo,
                pct_low=float(qlo), pct_high=float(qhi), n_positions=int(len(d)),
                n_models=int(d.model.nunique()), n_clusters=int(d.cluster.nunique()))


def main() -> int:
    rng = np.random.default_rng(SEED)
    res = {"m_star": m_star, "model_mean_on_off": model_mean.round(6).to_dict()}
    r = satterthwaite(hh1, "model", rng)
    F = floor_of(hh1, "model")
    r.update(floor=F, decision=decide(r, F, two_sided=True), models=sorted(hh1.model.unique()),
             snr_composition={"12,18": int(sum(edge_set(p) == [12.0, 18.0] for p in hh1.position_id)),
                              "18": int(sum(edge_set(p) == [18.0] for p in hh1.position_id))})
    res["HH1"] = r
    rn = satterthwaite(neg, "model", rng)
    rn.update(models=sorted(neg.model.unique()))
    res["HH1_negative_control"] = rn
    res["HH2"] = hh2_boot(hh2, rng)
    r = satterthwaite(hh3, "model", rng)
    F = floor_of(hh3, "model")
    r.update(floor=F, decision=decide(r, F), rej30_share_units_on_fold0=rej30_share_units,
             positions_missing_30db=pos30_missing, n_mstar_positions=int(len(mstar_pos)))
    res["HH3"] = r
    for k in ("HB1", "HB1b", "HB2"):
        r = satterthwaite(hb[k], "background", rng)
        F = floor_of(hb[k], "background")
        r.update(floor=F, decision=decide(r, F),
                 by_background={g: float(hb[k][hb[k].background == g].difference.mean()) for g in BGS})
        res[k] = r
    res["HH2_mstar"] = hh2_boot(hh2, rng, models=m_star)
    # by-background for model-weighted
    for k, df in (("HH1", hh1), ("HH3", hh3)):
        res[k]["by_background"] = {g: est_model(df[df.background == g].reset_index(drop=True)) for g in BGS}
    # HH1 boundary / snr composition and gray / unshared, quick point values
    res["counts"] = dict(
        hyperbola_positions=int(len(pos_info)),
        hyperbola_positions_both_arms=int(len(both)),
        hh2_positions=int(len(hh2)), hh2_models=int(hh2.model.nunique()),
        off_rejected_units_fold0=int(len(off_rej)),
        off_rejected_positions=sorted(off_rej.position_id.unique().tolist()),
        strata_all=pos_info.stratum.value_counts().to_dict(),
        strata_mstar=pos_info[pos_info.model.isin(m_star)].stratum.value_counts().to_dict(),
    )
    # compare with judge units csv (difference / floor per position)
    cmp = {}
    for name, df in (("HH1", hh1), ("HH1_negative_control", neg), ("HH3", hh3),
                     ("HB1", hb["HB1"]), ("HB1b", hb["HB1b"]), ("HB2", hb["HB2"])):
        j = pd.read_csv(JDG / f"units_{name}.csv", dtype={"cluster": str}).set_index("position_id")
        mine = df.set_index("position_id")
        same_set = set(j.index) == set(mine.index)
        dd = (j.difference - mine.difference.reindex(j.index)).abs().max() if same_set else None
        ff = (j.floor_value - mine.floor_value.reindex(j.index)).abs().max() if same_set else None
        cmp[name] = dict(same_positions=same_set, n_judge=int(len(j)), n_mine=int(len(mine)),
                         max_abs_diff_difference=None if dd is None else float(dd),
                         max_abs_diff_floor=None if ff is None else float(ff))
    j2 = pd.read_csv(JDG / "units_HH2.csv").set_index("position_id")
    cmp["HH2"] = dict(same_positions=set(j2.index) == set(hh2.index), n_judge=int(len(j2)),
                      max_abs_diff_value=float((j2.value - hh2.Y_on.reindex(j2.index)).abs().max()))
    res["units_csv_comparison"] = cmp
    # hh2 missing positions
    res["hh2_missing_positions"] = sorted(set(pos_info.index) - set(hp.dropna(subset=["Y_on"]).index))
    OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(OUT.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
