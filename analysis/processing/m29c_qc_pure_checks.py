"""Expected values of the pure-function QC tests C5b, C6b, C7(c), C8, C9, C10 (spec 9-2-2).

Uses the prototype functions of m29c_qc_proto.py (synthetic only, no 4TU data).
Run: conda run -n gpr-dev python -I analysis/processing/m29c_qc_pure_checks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m29c_qc_proto as proto  # noqa: E402

proto.RECORD_END_PERIODS = 2.5
HALF = proto.ref_half_traces(0.02)


def score(lag, amp):
    dj, aj, dl, al = proto.seam_from_track(lag, amp, HALF)
    with np.errstate(invalid="ignore"):
        return np.maximum(dj / dl, aj / al)


def c6b() -> None:
    n = 401
    base_l, base_a = np.zeros(n), np.ones(n)
    cases = {}
    lag = base_l.copy(); lag[200:] += 0.21; cases["i +0.21 ns"] = (lag, base_a)
    lag = base_l.copy(); lag[200:] += 0.19; cases["ii +0.19 ns"] = (lag, base_a)
    amp = base_a.copy(); amp[200:] *= 1.26; cases["iii x1.26"] = (base_l, amp)
    amp = base_a.copy(); amp[200:] *= 1.24; cases["iii x1.24"] = (base_l, amp)
    lag = np.random.default_rng(0).normal(0.0, 0.05, n); lag[200:] += 0.21
    cases["iv noise 0.05 + 0.21"] = (lag, base_a)
    amp = base_a.copy(); amp[200] = 0.0; cases["v zero amp at 200"] = (base_l, amp)
    for name, (lag, amp) in cases.items():
        s = score(lag, amp)
        others = np.delete(s, [198, 199, 200])
        print(f"C6b {name}: s[199]={s[199]:.4f} s[200]={s[200]:.4f} max_other={np.nanmax(others):.4f} n_seam={int(np.sum(s >= 1))}")


def c5b() -> None:
    for dt in (0.098, 0.117, 0.195):
        out = []
        for m in (5.0 - 2e-9, 5.0, 5.0 + 1e-6):
            sample = 511 - m / dt
            margin = (511 - sample) * dt
            out.append(margin < proto.RECORD_END_PERIODS * proto.PERIOD_NS - 1e-9)
        print("C5b", dt, out)


def fake(v, x0=0.0, t0=10.0, inlier=1.0, sigma=0.001, converged=True):
    fit = SimpleNamespace(v_m_per_ns=v, x0_m=x0, t0_ns=t0, sigma_v=sigma, inlier_frac=inlier,
                          converged=converged)
    curv = SimpleNamespace(status="fit_failed", fit=fit)
    x = np.linspace(-0.5, 0.5, 51)
    tau = 10.0 + np.abs(x)
    picks = SimpleNamespace(apex=object(), n_picks=51, n_left=25, n_right=25, x_m=x, tau_ns=tau,
                            apex_x_m=0.0)
    return curv, picks


def c7_c8_c9() -> None:
    vm = 0.1
    rows = {
        "not converged": fake(0.1, converged=False),
        "x0 at min": fake(0.1, x0=-0.5),
        "inlier 0.49": fake(0.1, inlier=0.49),
        "sigma NaN": fake(0.1, sigma=math.nan),
        "sigma 0.016 (0.16)": fake(0.1, sigma=0.016),
        "sigma 0.0149 (0.149)": fake(0.1, sigma=0.0149),
        "r 0.69": fake(0.069),
        "r 1.61": fake(0.161),
        "v upper bound": fake(0.2998),
        "v upper bound, x0 at bound": fake(0.2998, x0=0.5),
        "v lower bound": fake(0.03),
        "v upper - 1e-3 (r 2.988)": fake(0.2988),
    }
    for name, (curv, picks) in rows.items():
        q = proto.classify_curvature(curv, picks, vm)
        print(f"C7/C9 {name}: {q.verdict} {q.reason}")
    for r in (0.7, 1.6, 0.7 - 1e-13, 1.6 + 1e-13, 0.7 - 1e-9, 1.6 + 1e-9):
        print("C8", repr(r), proto.classify_ratio(r))
    for r in (math.nan, math.inf):
        try:
            proto.classify_ratio(r)
        except ValueError:
            print("C8", r, "ValueError")


def c10() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import eval_depth_4tu as ev

    depths = [0.60, 0.61, 0.62, 1.20, 1.21]
    lines = ["L1", "L2", "L3", "L1", "L2"]
    excluded = [False, True, False, True, False]
    for name, use in (("main", [not e for e in excluded]), ("noqc", [True] * 5)):
        d = [x for x, u in zip(depths, use, strict=True) if u]
        ln = [x for x, u in zip(lines, use, strict=True) if u]
        cl = ev.cluster_depths(d, ln, width_m=0.10, linkage="complete")
        print("C10", name, [(round(c.depth_m, 4), c.n_lines) for c in cl])


if __name__ == "__main__":
    c5b()
    c6b()
    c7_c8_c9()
    c10()
