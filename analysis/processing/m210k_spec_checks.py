"""M2-10 K1/K2 specification checks (synthetic tables only, no 4TU data).

Backs spec_eval_4tu_v1 v1.9 sections 5-2 (K2 tie range), 4-6 (K1 draft) with
numbers that python-expert's tests can reuse:

* K2 - pass-count range over co-optimal ungated L1 assignments.
  ``lambda`` method (one linear assignment each for min and max, perturbation
  lambda = TIE_COST_TOL_M / (m + 1)) against brute force over every injective
  assignment of size m. Random cases on a 0.01 m / 0.05 m depth grid
  (exact ties in real arithmetic), seed 20261111.
* K1 - prototype of the multiplicity-aware estimate construction (lower median
  of per-line lateral multiplicity over the cluster's non-long lines) on the S1
  cases, and the binomial false/true multiplicity tables (S1m).

Imports ``scripts/eval_depth_4tu.py`` read-only (cluster_depths, supported,
match_activity, match_activity_gated, _within_pass_limit). Writes
``docs/processing/m210k_spec_checks.json``.

Run (single process, BelowNormal, one core):
    conda run -n gpr-dev python -I analysis/processing/m210k_spec_checks.py
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10,
# same setting as scripts/eval_depth_4tu.py); an explicit value wins.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_depth_4tu as ev  # noqa: E402

OUT = ROOT / "docs" / "processing" / "m210k_spec_checks.json"
SEED = 20261111
TIE_COST_TOL_M = 1e-9
K1_LATERAL_M = 0.30  # generator NMS distance (fourtu_apex_candidate_select.NMS_M)
K1_LATERAL_TOL_M = 1e-9


# ------------------------------------------------------------------ K2


def pass_matrix(est, gt):
    cost = np.abs(np.subtract.outer(np.asarray(est, float), np.asarray(gt, float)))
    return cost, np.asarray(ev._within_pass_limit(cost), dtype=float)


def pass_range_lambda(est, gt, tol=TIE_COST_TOL_M):
    """(official, lo, hi) pass counts; lambda = tol / (m + 1)."""
    if len(est) == 0 or len(gt) == 0:
        return 0, 0, 0
    cost, passes = pass_matrix(est, gt)
    m = min(cost.shape)
    lam = tol / (m + 1)
    official = ev.match_activity(est, gt)
    n_off = sum(bool(ev._within_pass_limit(abs(e))) for _, _, e in official.pairs)
    r, c = linear_sum_assignment(cost - lam * passes)
    hi = int(passes[r, c].sum())
    s_hi = float(cost[r, c].sum())
    r, c = linear_sum_assignment(cost + lam * passes)
    lo = int(passes[r, c].sum())
    s_lo = float(cost[r, c].sum())
    r, c = linear_sum_assignment(cost)
    s_star = float(cost[r, c].sum())
    assert s_hi <= s_star + tol and s_lo <= s_star + tol, (s_hi, s_lo, s_star)
    return n_off, lo, hi


def pass_range_brute(est, gt, tol=TIE_COST_TOL_M):
    if len(est) == 0 or len(gt) == 0:
        return 0, 0
    cost, passes = pass_matrix(est, gt)
    ne, ng = cost.shape
    m = min(ne, ng)
    best = math.inf
    scored = []
    if ne <= ng:
        for cols in itertools.permutations(range(ng), m):
            rows = range(ne)
            s = sum(cost[i, j] for i, j in zip(rows, cols))
            k = int(sum(passes[i, j] for i, j in zip(rows, cols)))
            scored.append((s, k))
            best = min(best, s)
    else:
        for rows in itertools.permutations(range(ne), m):
            cols = range(ng)
            s = sum(cost[i, j] for i, j in zip(rows, cols))
            k = int(sum(passes[i, j] for i, j in zip(rows, cols)))
            scored.append((s, k))
            best = min(best, s)
    ks = [k for s, k in scored if s <= best + tol]
    return min(ks), max(ks)


def k2_examples():
    cases = {
        "S2-1_diagnosis_table": ([0.40, 0.45, 0.15], [0.42, 1.20, 1.25]),
        "S2-2_unique_optimum": ([0.50, 1.00], [0.52, 1.10]),
        "S2-8_float_tie": ([0.10, 0.12], [0.41, 0.50]),
        "S2-8b_float_tie_reversed": ([0.12, 0.10], [0.50, 0.41]),
        "S2-5_empty_est": ([], [0.5, 0.7]),
        "S2-5_empty_gt": ([0.5], []),
        "S2-1b_all_shallow": ([0.20, 0.25, 0.30], [0.50, 0.55, 0.95, 1.40]),
    }
    out = {}
    for name, (e, g) in cases.items():
        n_off, lo, hi = pass_range_lambda(e, g)
        blo, bhi = pass_range_brute(e, g)
        gate = ev.match_activity_gated(e, g).n_pairs if e and g else 0
        cost = np.abs(np.subtract.outer(np.asarray(e, float), np.asarray(g, float))) if e and g else None
        s_star = None
        if cost is not None:
            r, c = linear_sum_assignment(cost)
            s_star = float(cost[r, c].sum())
        out[name] = {
            "est": e, "gt": g, "official": n_off, "lo": lo, "hi": hi,
            "brute_lo": blo, "brute_hi": bhi, "gate_pairs": gate, "s_star_m": s_star,
        }
    return out


def k2_random(n_cases=3000):
    rng = np.random.default_rng(SEED)
    mism = 0
    order_mism = 0
    prop_fail = 0
    n_range = 0
    width_hist = {}
    for case in range(n_cases):
        grid = 0.01 if case % 2 == 0 else 0.05
        lo_d, hi_d = (0.1, 2.0) if case % 2 == 0 else (0.2, 1.2)
        ne, ng = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        e = list(np.round(rng.uniform(lo_d, hi_d, ne) / grid) * grid)
        g = list(np.round(rng.uniform(lo_d, hi_d, ng) / grid) * grid)
        n_off, lo, hi = pass_range_lambda(e, g)
        blo, bhi = pass_range_brute(e, g)
        if (lo, hi) != (blo, bhi):
            mism += 1
        gate = ev.match_activity_gated(e, g).n_pairs if e and g else 0
        if not (lo <= n_off <= hi <= gate):
            prop_fail += 1
        pe = list(rng.permutation(e)) if e else []
        pg = list(rng.permutation(g)) if g else []
        _, lo2, hi2 = pass_range_lambda(pe, pg)
        if (lo2, hi2) != (lo, hi):
            order_mism += 1
        if hi > lo:
            n_range += 1
        width_hist[hi - lo] = width_hist.get(hi - lo, 0) + 1
    return {
        "seed": SEED,
        "cases": n_cases,
        "grids_m": [0.01, 0.05],
        "lambda_vs_brute_mismatch": mism,
        "order_invariance_mismatch": order_mism,
        "property_lo_le_official_le_hi_le_gate_fail": prop_fail,
        "cases_with_range": n_range,
        "range_width_histogram": {str(k): v for k, v in sorted(width_hist.items())},
    }


# ------------------------------------------------------------------ K1 prototype


def lateral_multiplicity(xs):
    """Single-linkage groups on x with link if gap <= 0.30 m (+1e-9)."""
    xs = sorted(xs)
    if not xs:
        return 0
    groups = 1
    for a, b in zip(xs, xs[1:]):
        if b - a > K1_LATERAL_M + K1_LATERAL_TOL_M:
            groups += 1
    return groups


def k1_estimates(depths, lines, xs, long_lines, n_lines_activity,
                 width_m=ev.CLUSTER_WIDTH_M, min_lines=ev.MIN_SUPPORT_LINES):
    clusters = ev.cluster_depths(depths, lines, width_m, "complete")
    kept = ev.supported(clusters, n_lines_activity, min_lines)
    base = [c.depth_m for c in kept]
    out = []
    counts = []
    for c in kept:
        per_line = {}
        for i in c.member_indices:
            if lines[i] in long_lines:
                continue
            per_line.setdefault(lines[i], []).append(xs[i])
        mus = sorted(lateral_multiplicity(v) for v in per_line.values())
        n_c = 1 if not mus else mus[(len(mus) - 1) // 2]
        counts.append(n_c)
        out.extend([c.depth_m] * n_c)
    return base, out, counts


def s1_cases():
    rng = np.random.default_rng(SEED + 1)
    res = {}

    def run(name, rows, long_lines=(), n_lines=None):
        d = [r[0] for r in rows]
        ln = [r[1] for r in rows]
        x = [r[2] for r in rows]
        n_lines = n_lines if n_lines is not None else len(set(ln))
        base, k1, counts = k1_estimates(d, ln, x, set(long_lines), n_lines)
        # permutation invariance
        perm = rng.permutation(len(rows))
        b2, k2, c2 = k1_estimates([d[i] for i in perm], [ln[i] for i in perm],
                                  [x[i] for i in perm], set(long_lines), n_lines)
        res[name] = {
            "n_current": len(base), "n_k1": len(k1), "counts": counts,
            "k1_depths": [round(v, 6) for v in k1],
            "permutation_equal": (b2 == base and k2 == k1 and c2 == counts),
        }

    # S1-1: k pipes at 0.60 m, lateral 0.6-1.5 m, 6 lines, sigma 0.02 m
    for k in (2, 3):
        rows = []
        for li in range(6):
            x0 = 2.0
            for p in range(k):
                x0 += [0.6, 1.0][p - 1] if p else 0.0
                rows.append((0.60 + rng.normal(0, 0.02), f"L{li}", x0 + rng.normal(0, 0.02)))
        run(f"S1-1_k{k}", rows)
    # S1-2: one pipe per line (null)
    rows = [(0.60 + rng.normal(0, 0.02), f"L{li}", 3.0) for li in range(6)]
    rows += [(1.10 + rng.normal(0, 0.02), f"L{li}", 5.0) for li in range(6)]
    run("S1-2_null_two_depths", rows)
    # S1-3: converged duplicate apexes (dx <= 0.30) -> mu 1
    rows = []
    for li in range(4):
        rows += [(0.60, f"L{li}", 3.00), (0.61, f"L{li}", 3.30)]
    run("S1-3_converged_0.30", rows)
    rows = []
    for li in range(4):
        rows += [(0.60, f"L{li}", 3.00), (0.61, f"L{li}", 3.30 + 2e-9)]
    run("S1-3b_just_beyond_0.30", rows)
    # S1-4: zigzag long line, single-line activity, same pipe 3 times
    rows = [(0.60, "Z", 5.0), (0.61, "Z", 15.0), (0.60, "Z", 25.0)]
    run("S1-4_long_line_only", rows, long_lines=("Z",))
    run("S1-4b_same_not_long", rows)
    # S1-5: lower median rounding
    run("S1-5_mu_2_1", [(0.6, "A", 1.0), (0.6, "A", 2.0), (0.6, "B", 1.0)])
    run("S1-5_mu_2_2_1", [(0.6, "A", 1.0), (0.6, "A", 2.0), (0.6, "B", 1.0),
                          (0.6, "B", 2.0), (0.6, "C", 1.0)])
    run("S1-5_mu_3_3_2_2", [(0.6, L, x) for L, xs in
                            (("A", (1, 2, 3)), ("B", (1, 2, 3)), ("C", (1, 2)), ("D", (1, 2)))
                            for x in xs])
    # S1-6: second pipe on 2 of 6 lines -> no increase
    rows = [(0.60, f"L{li}", 3.0) for li in range(6)]
    rows += [(0.61, f"L{li}", 4.0) for li in range(2)]
    run("S1-6_minority", rows)
    return res


def s1m_tables():
    ns = list(range(1, 13))
    ps = [0.05, 0.1, 0.2, 0.3, 0.5]
    qs = [0.5, 0.7, 0.9, 1.0]

    def need(n):
        return n // 2 + 1

    false_k1 = {str(p): [round(float(binom.sf(need(n) - 1, n, p)), 4) for n in ns] for p in ps}
    false_support = {
        str(p): [round(float(p if n == 1 else binom.sf(1, n, p)), 4) for n in ns] for p in ps
    }
    recov = {}
    for q in qs:
        for p in (0.0, 0.1, 0.3):
            qq = 1 - (1 - q) * (1 - p)
            recov[f"q{q}_p{p}"] = [round(float(binom.sf(need(n) - 1, n, qq)), 4) for n in ns]
    dominated = all(
        false_k1[str(p)][i] <= false_support[str(p)][i] + 1e-12 for p in ps for i in range(len(ns))
    )
    # Monte Carlo check of the closed form with the prototype
    rng = np.random.default_rng(SEED + 2)
    mc = {}
    for n in (2, 3, 6):
        for p in (0.1, 0.3):
            hits = 0
            trials = 4000
            for _ in range(trials):
                rows = [(0.60, f"L{li}", 3.0) for li in range(n)]
                for li in range(n):
                    if rng.random() < p:
                        rows.append((0.60 + rng.uniform(-0.04, 0.04), f"L{li}",
                                     3.0 + rng.choice([-1, 1]) * rng.uniform(0.35, 2.0)))
                d = [r[0] for r in rows]
                ln = [r[1] for r in rows]
                x = [r[2] for r in rows]
                _, k1, _ = k1_estimates(d, ln, x, set(), n)
                base, _, _ = k1_estimates(d, ln, x, set(), n)
                hits += len(k1) > len(base)
            mc[f"n{n}_p{p}"] = {"mc": round(hits / trials, 4),
                                "closed_form": false_k1[str(p)][n - 1], "trials": trials}
    return {
        "n_lines": ns,
        "rule": "extra estimate iff at least floor(n/2)+1 of the cluster's non-long lines show mu >= 2",
        "false_extra_k1": false_k1,
        "false_supported_cluster_current": false_support,
        "k1_never_admits_clutter_more_easily_than_support_filter": dominated,
        "recovery_two_pipes": recov,
        "mc_check_seed": SEED + 2,
        "mc_check": mc,
    }


def main():
    out = {
        "note": "synthetic tables only; no 4TU data read",
        "tie_cost_tol_m": TIE_COST_TOL_M,
        "k2_examples": k2_examples(),
        "k2_random": k2_random(),
        "k1_lateral_m": K1_LATERAL_M,
        "s1_cases": s1_cases(),
        "s1m": s1m_tables(),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "k2_random": out["k2_random"],
        "k2_examples": {k: (v["official"], v["lo"], v["hi"], v["brute_lo"], v["brute_hi"], v["gate_pairs"])
                        for k, v in out["k2_examples"].items()},
        "s1": {k: (v["n_current"], v["n_k1"], v["counts"], v["permutation_equal"])
               for k, v in out["s1_cases"].items()},
        "dominated": out["s1m"]["k1_never_admits_clutter_more_easily_than_support_filter"],
        "mc": out["s1m"]["mc_check"],
    }))


if __name__ == "__main__":
    main()
