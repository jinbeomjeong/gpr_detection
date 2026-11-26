"""GPR-SD M3-3b U10: re-check AR1-CR1 independence (plan docs/experiments/m33b_gprsd_plan.md 6, 11 U10).

Uses only marks.csv (type, trace position), radargrams_index.csv, and block-mean log|amp| profiles (1-D).
No pipe/hyperbola detection, no label windows, no in-window apex (blind, plan Q8). Read-only on the dataset.

Run: conda run -n gpr-dev python -I analysis/data/gprsd_m33b_ar1_cr1_check.py
Output: data/labels/gprsd_m34/road_independence_ar1_cr1.json (git-ignored) + stdout summary.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PKG = Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016")
SEED = 20261126
BLOCK = 50
MAXLAG = 120
NSIM = 5000


def main() -> int:
    with open(PKG / "01_메타데이터" / "marks.csv", encoding="utf-8-sig", newline="") as f:
        marks = list(csv.DictReader(f))
    with open(PKG / "01_메타데이터" / "radargrams_index.csv", encoding="utf-8-sig", newline="") as f:
        index = {r["file_id"]: r for r in csv.DictReader(f)}
    roads = sorted({m["road"] for m in marks})
    pos = {r: sorted((int(m["location_trace"]), m["type"]) for m in marks if m["road"] == r) for r in roads}
    L = {r: min(int(v["n_traces"]) for v in index.values() if v["road"] == r) for r in roads}
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED, "nsim": NSIM}

    # A. mark granularity
    gran = {}
    for r in roads:
        t = [p[0] for p in pos[r]]
        gran[r] = {"n": len(t), "mult_of_100": sum(x % 100 == 0 for x in t), "mult_of_500": sum(x % 500 == 0 for x in t),
                   "min_trace": min(t), "max_trace": max(t), "line_len": L[r]}
    out["mark_granularity"] = gran

    # B. coincidence null with 100-trace shift granularity (marks are rounded to 100)
    types_all = Counter(p[1] for r in roads for p in pos[r])
    out["type_counts_all_roads"] = dict(types_all)
    res = {}
    for a, b in itertools.combinations(roads, 2):
        ta = [p[0] for p in pos[a]]
        tb = np.array([p[0] for p in pos[b]])
        typ_b = [p[1] for p in pos[b]]
        Lm = min(L[a], L[b])
        sa = set(ta)
        coin = [(x, ta_t, tb_t) for (x, ta_t) in pos[a] for (y, tb_t) in pos[b] if x == y]
        obs = len(coin)
        same = sum(c[1] == c[2] for c in coin)
        nulls, nulls_same = [], []
        for _ in range(NSIM):
            sh = int(rng.integers(0, Lm // 100)) * 100
            tb_s = (tb + sh) % Lm
            hit = [(int(x), t) for x, t in zip(tb_s.tolist(), typ_b) if x in sa]
            nulls.append(len(hit))
            tad = dict(pos[a])
            nulls_same.append(sum(tad[x] == t for x, t in hit))
        nulls = np.array(nulls)
        nulls_same = np.array(nulls_same)
        res[f"{a}|{b}"] = {
            "obs": obs, "obs_same_type": same,
            "null100_mean": float(nulls.mean()), "null100_p95": float(np.percentile(nulls, 95)),
            "p_ge_obs_null100": float((nulls >= obs).mean()),
            "null100_same_type_mean": float(nulls_same.mean()),
            "p_ge_obs_same_type_null100": float((nulls_same >= same).mean()),
            "pipe_pipe_coincidences": sum(c[1] == "pipeline" and c[2] == "pipeline" for c in coin),
            "coin": [(c[0], c[1], c[2]) for c in coin],
        }
    out["mark_coincidence_null100"] = res

    # C. profiles: block-mean log|amp|, high-passed (21-block moving mean removed)
    prof = {}
    for fid in sorted(index):
        a = np.load(PKG / "05_radargrams_npy" / f"{fid}.npy", mmap_mode="r")
        n = (a.shape[1] // BLOCK) * BLOCK
        m = np.abs(np.asarray(a[:, :n])).reshape(a.shape[0], n // BLOCK, BLOCK).mean(axis=(0, 2))
        prof[fid] = np.log(m + 1e-6)

    def hp(x):
        k = np.ones(21) / 21
        return x - np.convolve(np.pad(x, 10, mode="edge"), k, mode="valid")

    def corr(x, y, lag):
        if lag >= 0:
            x2, y2 = x[lag:], (y[: len(y) - lag] if lag else y)
        else:
            x2, y2 = x[: len(x) + lag], y[-lag:]
        n = min(len(x2), len(y2))
        if n < 50:
            return float("nan")
        return float(np.corrcoef(x2[:n], y2[:n])[0, 1])

    hpd = {f: hp(prof[f]) for f in prof}
    road_of = {f: index[f]["road"] for f in index}
    rows = {}
    for key in ("AsphaltRoad1|CementRoad1", "AsphaltRoad1|AsphaltRoad2", "AsphaltRoad1|CementRoad2", "AsphaltRoad2|CementRoad1", "CementRoad1|CementRoad2"):
        ra, rb = key.split("|")
        lst = []
        for fa in sorted(f for f in index if road_of[f] == ra):
            for fb in sorted(f for f in index if road_of[f] == rb):
                lags = [(corr(hpd[fa], hpd[fb], g), g) for g in range(-MAXLAG, MAXLAG + 1)]
                bc, bl = max(lags, key=lambda t: np.nan_to_num(t[0], nan=-9))
                lst.append({"a": fa, "b": fb, "hp_lag0": corr(hpd[fa], hpd[fb], 0), "hp_best": bc, "hp_best_lag": bl,
                            "raw_lag0": corr(prof[fa], prof[fb], 0)})
        rows[key] = lst
    out["file_pair_profile"] = rows

    # D. best-lag corr significance for AR1-CR1 vs circular-shift null (hp profiles, mean over file pairs, same-lag window)
    def best_over_lags(x, y):
        return max(np.nan_to_num(corr(x, y, g), nan=-9) for g in range(-MAXLAG, MAXLAG + 1))

    sig = {}
    for key in ("AsphaltRoad1|CementRoad1", "AsphaltRoad1|AsphaltRoad2"):
        ra, rb = key.split("|")
        fa, fb = sorted(f for f in index if road_of[f] == ra)[0], sorted(f for f in index if road_of[f] == rb)[0]
        obs = best_over_lags(hpd[fa], hpd[fb])
        nl = []
        for _ in range(200):
            sh = int(rng.integers(130, len(hpd[fb]) - 130))
            nl.append(best_over_lags(hpd[fa], np.roll(hpd[fb], sh)))
        nl = np.array(nl)
        sig[key] = {"files": [fa, fb], "obs_best_hp_corr": obs, "null_roll_mean": float(nl.mean()), "null_roll_p95": float(np.percentile(nl, 95)),
                    "null_roll_max": float(nl.max()), "p_ge_obs": float((nl >= obs).mean())}
    out["best_lag_significance_roll_null"] = sig

    p = REPO / "data" / "labels" / "gprsd_m34" / "road_independence_ar1_cr1.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("mark_granularity", "type_counts_all_roads", "best_lag_significance_roll_null")}, ensure_ascii=False, indent=1))
    for k, v in res.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for k, lst in rows.items():
        print(k, [(round(r["hp_lag0"], 2), round(r["hp_best"], 2), r["hp_best_lag"], round(r["raw_lag0"], 2)) for r in lst])
    return 0


if __name__ == "__main__":
    sys.exit(main())
