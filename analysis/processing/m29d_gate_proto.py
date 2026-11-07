"""M2-9 improvement D prototype: +-0.30 m gated matching (spec_eval_4tu_v1 v1.6, 5-1).

Checks, without any 4TU data:

1. ``gated_match`` (linear assignment with a finite penalty ``P = m * g + 1`` for
   pairs outside the gate) equals the lexicographic optimum (max pairs, then
   min sum |e|) found by brute force, on random small activities whose depths
   lie on a 0.05 m grid (exact ties and float boundary values like
   0.45 - 0.15 = 0.30000000000000004 occur on purpose).
2. ``gate_pairs >= n_pass`` of the ungated Hungarian (same predicate).
3. The hand fixtures G1-G5 and G7 of spec 9-2-3 and two mutations per fixture.

Writes ``docs/processing/m29d_gate_proto.json``. Seed 20261107.
Run:  conda run -n gpr-dev python -I analysis/processing/m29d_gate_proto.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "processing" / "m29d_gate_proto.json"
GATE_M = 0.30
TOL_M = 1e-12
SEED = 20261107
N_CASES = 3000
MAX_N = 4
GRID_STEP_M = 0.05
GRID_MAX_K = 40


def within(abs_error: float) -> bool:
    return abs_error <= GATE_M + TOL_M


def ungated_n_pass(est, gt) -> int:
    if not est or not gt:
        return 0
    cost = np.abs(np.subtract.outer(np.asarray(est, float), np.asarray(gt, float)))
    rows, cols = linear_sum_assignment(cost)
    return sum(within(cost[r, c]) for r, c in zip(rows, cols, strict=True))


def gated_match(est, gt, penalty_mode="spec", admissible_cost="abs"):
    """Return (k, sum|e|). penalty_mode/admissible_cost allow mutations."""
    if not est or not gt:
        return 0, 0.0
    abs_err = np.abs(np.subtract.outer(np.asarray(est, float), np.asarray(gt, float)))
    ok = abs_err <= GATE_M + TOL_M
    m = min(abs_err.shape)
    penalty = {"spec": m * GATE_M + 1.0, "small": GATE_M, "none": None}[penalty_mode]
    base = abs_err if admissible_cost == "abs" else np.zeros_like(abs_err)
    cost = base.copy() if penalty is None else np.where(ok, base, penalty)
    if penalty is None:
        cost = abs_err
    rows, cols = linear_sum_assignment(cost)
    keep = [(r, c) for r, c in zip(rows, cols, strict=True) if ok[r, c]]
    return len(keep), float(sum(abs_err[r, c] for r, c in keep))


def brute_force(est, gt):
    """Lexicographic optimum over all partial matchings inside the gate."""
    best = (0, 0.0)
    n_e, n_g = len(est), len(gt)
    for k in range(min(n_e, n_g), 0, -1):
        sums = []
        for rows in itertools.combinations(range(n_e), k):
            for cols in itertools.permutations(range(n_g), k):
                errors = [abs(est[r] - gt[c]) for r, c in zip(rows, cols, strict=True)]
                if all(within(e) for e in errors):
                    sums.append(sum(errors))
        if sums:
            return k, min(sums)
    return best


def random_check() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    n_k_mismatch = n_sum_mismatch = n_relation_fail = n_strict = n_boundary = 0
    max_sum_diff = 0.0
    for _ in range(N_CASES):
        n_e, n_g = rng.integers(0, MAX_N + 1, size=2)
        est = [float(k) * GRID_STEP_M for k in rng.integers(0, GRID_MAX_K + 1, n_e)]
        gt = [float(k) * GRID_STEP_M for k in rng.integers(0, GRID_MAX_K + 1, n_g)]
        k_lsa, s_lsa = gated_match(est, gt)
        k_bf, s_bf = brute_force(est, gt)
        n_pass = ungated_n_pass(est, gt)
        n_k_mismatch += k_lsa != k_bf
        diff = abs(s_lsa - s_bf)
        max_sum_diff = max(max_sum_diff, diff)
        n_sum_mismatch += diff > 1e-9
        n_relation_fail += k_lsa < n_pass
        n_strict += k_lsa > n_pass
        n_boundary += any(
            abs(abs(e - g) - GATE_M) < 1e-9 for e in est for g in gt
        )
    return {
        "cases": N_CASES,
        "seed": SEED,
        "k_mismatch": n_k_mismatch,
        "sum_mismatch_gt_1e-9": n_sum_mismatch,
        "max_sum_abs_diff": max_sum_diff,
        "gate_pairs_lt_n_pass": n_relation_fail,
        "gate_pairs_gt_n_pass_cases": n_strict,
        "cases_with_boundary_pair": n_boundary,
    }


def fixtures() -> dict[str, object]:
    g1_est, g1_gt = [0.45, 0.85], [0.15, 0.60, 1.20]
    g2_est, g2_gt = [0.5, 0.6], [0.75, 0.55]
    out = {
        "g1_boundary_value": 0.45 - 0.15,
        "g1_boundary_gt_0.30": (0.45 - 0.15) > GATE_M,
        "g1_ungated_n_pass": ungated_n_pass(g1_est, g1_gt),
        "g1_gated": gated_match(g1_est, g1_gt),
        "g1_mut_small_penalty": gated_match(g1_est, g1_gt, penalty_mode="small"),
        "g1_mut_no_gate": gated_match(g1_est, g1_gt, penalty_mode="none"),
        "g1_mut_no_tolerance_k": sum(
            1 for e, g in [(0.45, 0.15), (0.85, 0.60)] if abs(e - g) <= GATE_M
        ),
        "g2_gated": gated_match(g2_est, g2_gt),
        "g2_mut_zero_cost": gated_match(g2_est, g2_gt, admissible_cost="zero"),
        "g2_ungated_n_pass": ungated_n_pass(g2_est, g2_gt),
    }
    # G3: micro aggregation over A (est 3, gt 1) and B (est 0, gt 2).
    k_a, _ = gated_match([1.0, 1.1, 1.2], [1.05])
    k_b, _ = gated_match([], [0.5, 0.9])
    k, n_gt, n_est = k_a + k_b, 3, 3
    out["g3_micro"] = {
        "k": k,
        "recall": k / n_gt,
        "precision": k / n_est,
        "f1": 2 * k / (n_gt + n_est),
    }
    out["g3_macro_recall_mutation"] = (k_a / 1 + k_b / 2) / 2
    out["g3_precision_over_pairs_mutation"] = k / (1 + 0)
    # G5: no estimate anywhere.
    out["g5"] = {"k": 0, "recall": 0.0, "precision": None, "f1": 0.0}
    # G7: baseline position.
    k_b_trials = np.array([0, 1, 1, 2, 3])
    k_obs = 1
    out["g7"] = {
        "p": float(np.mean(k_b_trials >= k_obs)),
        "pct": float(
            100 * (np.mean(k_b_trials < k_obs) + 0.5 * np.mean(k_b_trials == k_obs))
        ),
        "p_strict_mutation": float(np.mean(k_b_trials > k_obs)),
        "pct_no_midrank_mutation": float(100 * np.mean(k_b_trials < k_obs)),
    }
    return out


def _plain(value):
    """numpy scalars -> python scalars for json."""
    return value.item()


def main() -> int:
    result = {"random_check": random_check(), "fixtures": fixtures()}
    text = json.dumps(result, indent=2, ensure_ascii=True, default=_plain)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, default=_plain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
