"""Physics budget for the complete-linkage width of 4TU apex depths (M2-9 item B).

No 4TU radargram and no ground truth is read. Inputs are the spec uncertainty
budget (spec_depth_v1 section 9), equipment metadata (dt, f_c, eps_r_used from
docs/processing/fourtu_lines_v1.csv: per-activity uniqueness of eps_r_used and
the number of lines per activity) and analytic synthetic B-scans.

Parts
  A. Noise Monte Carlo of the real eval path (candidate_box -> estimate_depth
     with apex_hint) on air-coupled 500 MHz synthetic lines: repeat scatter of
     the pipe-top depth of ONE pipe over independent noise draws ("lines").
  B. Upper quantiles of the range of n iid normal values (w_n, Monte Carlo).
  C. Width table W = w_n * sigma_line(d) for budget variants.
  D. Prototype of the 1-D complete-linkage clustering (spec_eval_4tu_v1 v1.3
     section 4-3) on the synthetic test inputs B1-B12 -> expected values.

Run (single process, BLAS threads 1):
  conda run -n gpr-dev python -I analysis/processing/cluster_width_v1.py
Out:
  docs/processing/cluster_width_v1_noise_mc.csv   (part A)
  docs/processing/cluster_width_v1_range_q.csv    (part B)
  docs/processing/cluster_width_v1_table.csv      (part C)
  stdout: part D expected values and the eps_r_used uniqueness check
Seeds: numpy default_rng(0) for A (one stream in loop order), default_rng(1) for B,
default_rng(2) for D case permutations, default_rng(0) for the D property check B12.
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import csv  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from eval_depth_4tu import candidate_box, cluster_depths  # noqa: E402
from synthetic_bscan import T_SRC_NS, BScanSpec, make_radargram  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.picks import ApexHint  # noqa: E402
from gpr_detection.depth.refraction import reflection_times  # noqa: E402
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS  # noqa: E402

DOC = ROOT / "docs" / "processing"
C0 = SPEED_OF_LIGHT_M_PER_NS
F_C = 500e6
PERIOD_NS = 1e9 / F_C
AIR_GAP_M = 0.05  # 4TU-like single merged zero band (fourtu_explore: h <~ 0.1 m)
OFFSET_M = 0.1
N_TRACES = 151

DTS = (0.098, 0.195)
EPSS = (9.0, 16.0)
DEPTHS = (0.3, 0.6, 1.0, 1.5)
SNRS_DB = (20.0, 14.0)  # 14 dB = apex amplitude 5 sigma_n (hint floor k = 5)
N_DRAWS = 30

SIGMA_Z_NS = {"air_single_event": 0.25, "air_surface_late": 0.30}  # spec_depth 9
SIGMA_A_FLOOR_NS = 0.05  # spec_depth 9


# ----------------------------------------------------------------------------- A
def noise_mc() -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    config = DepthConfig()
    for dt in DTS:
        for eps in EPSS:
            v = C0 / math.sqrt(eps)
            for depth in DEPTHS:
                for snr in SNRS_DB:
                    n_samples = math.ceil((T_SRC_NS + 4 * depth / v + 15) / dt)
                    spec = BScanSpec(
                        eps_r=eps, d_top_m=depth, f_c_hz=F_C, air_gap_m=AIR_GAP_M,
                        offset_m=OFFSET_M, dt_ns=dt, n_samples=n_samples,
                        n_traces=N_TRACES, snr_db=None,
                    )
                    base_times = reflection_times(
                        spec.x_m, offset_m=OFFSET_M, h=AIR_GAP_M, x0_m=spec.x0_m,
                        d_top_m=depth, radius_m=0.0, v=v,
                    )
                    sigma_n = spec.pipe_amplitude / 10 ** (snr / 20)
                    errs, taus, zeros, n_fail = [], [], [], 0
                    for draw in range(N_DRAWS):
                        # sub-sample offset of the pipe event (grid phase), corrected in truth
                        u = float(rng.uniform(0.0, dt))
                        spec.pipe_times_ns = base_times + u
                        clean = make_radargram(spec).data
                        noisy = clean + rng.normal(0.0, sigma_n, clean.shape)
                        rg = make_radargram(
                            spec, data=noisy.astype(np.float32), eps_r_meta=eps,
                            line_id=f"mc.1/Path{draw}",
                        )
                        inputs = prepare_depth_inputs(rg, config)
                        tz = inputs.zero.t_zero_ns
                        apex = N_TRACES // 2
                        trace = apex + int(rng.integers(-2, 3))
                        sample = (T_SRC_NS + spec.pipe_times_ns[apex]) / dt + float(
                            rng.uniform(-0.25, 0.25)
                        ) * PERIOD_NS / dt
                        if tz is None:
                            n_fail += 1
                            continue
                        box = candidate_box(rg, trace, sample, sample * dt - tz)
                        res = estimate_depth(
                            rg, box, config, inputs=inputs,
                            apex_hint=ApexHint(trace=trace, sample=sample),
                        )
                        if res.status != "ok" or res.depth_top_m is None:
                            n_fail += 1
                            continue
                        errs.append(res.depth_top_m - (depth + v * u / 2))
                        taus.append(res.t_apex_ns - res.t_zero_ns - u)
                        zeros.append(res.t_zero_ns)
                    e = np.asarray(errs)
                    rows.append({
                        "dt_ns": dt, "eps_r": eps, "v_m_per_ns": round(v, 5),
                        "d_true_m": depth, "snr_db": snr, "n_draws": N_DRAWS,
                        "n_ok": len(e), "n_fail": n_fail,
                        "bias_m": round(float(e.mean()), 5) if len(e) else "",
                        "sd_depth_m": round(float(e.std(ddof=1)), 5) if len(e) > 1 else "",
                        "sd_tau_ns": round(float(np.std(taus, ddof=1)), 4) if len(e) > 1 else "",
                        "sd_t_zero_ns": round(float(np.std(zeros, ddof=1)), 4) if len(e) > 1 else "",
                        "p95_abs_dev_m": round(float(np.quantile(np.abs(e - e.mean()), 0.95)), 5)
                        if len(e) > 1 else "",
                    })
                    print("[A]", rows[-1], flush=True)
    return rows


# ----------------------------------------------------------------------------- B
def range_quantiles(ns=range(2, 13), draws=400_000) -> list[dict]:
    rng = np.random.default_rng(1)
    rows = []
    for n in ns:
        x = rng.standard_normal((draws, n))
        r = x.max(axis=1) - x.min(axis=1)
        rows.append({
            "n": n, "mean": round(float(r.mean()), 3),
            "q50": round(float(np.quantile(r, 0.5)), 3),
            "q90": round(float(np.quantile(r, 0.9)), 3),
            "q95": round(float(np.quantile(r, 0.95)), 3),
            "q99": round(float(np.quantile(r, 0.99)), 3),
        })
    return rows


# ----------------------------------------------------------------------------- C
def width_table(w_by_n: dict[int, float], sigma_a_ns: float) -> list[dict]:
    rows = []
    for eps in (8.16, 9.0, 19.46):
        v = C0 / math.sqrt(eps)
        for rule, sz in SIGMA_Z_NS.items():
            sigma_t = math.hypot(sz, sigma_a_ns)
            for sigma_g in (0.0, 0.01, 0.02, 0.03):
                for kappa in (0.0, 0.05, 0.15):
                    for depth in (0.3, 0.6, 1.0, 1.5, 2.0):
                        s_line = math.sqrt((v / 2 * sigma_t) ** 2 + sigma_g**2
                                           + (kappa * depth / 2) ** 2)
                        for n in (2, 6, 12):
                            rows.append({
                                "eps_r": eps, "v_m_per_ns": round(v, 4), "zero_rule": rule,
                                "sigma_t_ns": round(sigma_t, 4), "sigma_g_m": sigma_g,
                                "kappa_lat": kappa, "d_m": depth, "n_lines": n,
                                "sigma_line_m": round(s_line, 4),
                                "W_q95_m": round(w_by_n[n] * s_line, 4),
                            })
    return rows


# ----------------------------------------------------------------------------- D
TIE_TOL = 1e-12  # relative-ratio tie tolerance
EDGE_TOL_M = 1e-9  # inclusive width boundary


def width_fixed(w_m):
    return lambda mid: w_m


def complete_linkage(depths, lines, width_fn):
    """Prototype of spec 4-3 (v1.3): 1-D agglomerative complete linkage.

    Sorted by (depth, str(line)); clusters are contiguous runs; repeatedly merge
    the adjacent pair with the smallest ratio range / W(mid) among pairs with
    range <= W(mid) + EDGE_TOL_M; near-ties (ratio within TIE_TOL) -> shallower pair.
    Returns list of (median, n_lines, n_members, min, max, members).
    """
    n = len(depths)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (float(depths[i]), str(lines[i])))
    s = [float(depths[i]) for i in order]
    runs = [[k, k + 1] for k in range(n)]  # [start, end) in sorted order
    while len(runs) > 1:
        best, best_ratio = None, math.inf
        for j in range(len(runs) - 1):
            lo, hi = s[runs[j][0]], s[runs[j + 1][1] - 1]
            span, mid = hi - lo, 0.5 * (lo + hi)
            width = width_fn(mid)
            if span > width + EDGE_TOL_M:
                continue
            ratio = span / width
            if ratio < best_ratio - TIE_TOL:
                best, best_ratio = j, ratio
        if best is None:
            break
        runs[best] = [runs[best][0], runs[best + 1][1]]
        del runs[best + 1]
    out = []
    for a, b in runs:
        members = [order[k] for k in range(a, b)]
        out.append((
            round(float(np.median(s[a:b])), 6),
            len({str(lines[i]) for i in members}),
            b - a,
            round(s[a], 6),
            round(s[b - 1], 6),
            tuple(sorted((round(float(depths[i]), 6), str(lines[i])) for i in members)),
        ))
    return out


def summary(clusters):
    return [(c[0], c[1], c[2], round(c[4] - c[3], 6)) for c in clusters]


def prototype_cases() -> None:
    rng = np.random.default_rng(2)
    W = 0.10

    def show(tag, depths, lines, width_fn=width_fixed(W), old=True):
        new = complete_linkage(depths, lines, width_fn)
        print(f"[D] {tag} complete:", summary(new))
        if old:
            single = cluster_depths(list(depths), list(lines), 0.10, linkage="single")
            print(f"[D] {tag} single(0.10):",
                  [(round(c.depth_m, 6), c.n_lines, c.n_apex_candidates) for c in single])
        # order independence over 50 permutations
        ref = new
        for _ in range(50):
            p = rng.permutation(len(depths))
            got = complete_linkage([depths[i] for i in p], [lines[i] for i in p], width_fn)
            assert [c[:3] + c[5:] for c in got] == [c[:3] + c[5:] for c in ref], tag
        return new

    # B1 chain: 21 depths 0.30..0.90 step 0.03 over 6 lines
    d1 = [round(0.30 + 0.03 * k, 2) for k in range(21)]
    l1 = [f"L{k % 6 + 1}" for k in range(21)]
    show("B1 chain", d1, l1)
    # B1b chain like the 4TU defect: 6 lines x 8 candidates, step 0.02 interleaved
    d1b = [round(0.40 + 0.02 * k, 2) for k in range(48)]
    l1b = [f"L{k % 6 + 1}" for k in range(48)]
    show("B1b dense", d1b, l1b)
    # B4 boundary
    for pair in ([0.5, 0.6], [0.7, 0.8], [1.1, 1.2], [0.5, 0.6 + 2e-9]):
        print("[D] B4", pair, "float span", repr(pair[1] - pair[0]),
              summary(complete_linkage(pair, ["L1", "L2"], width_fixed(W))))
    # B5 two pipes outside width with a bridge
    d5 = [0.59, 0.60, 0.61, 0.60, 0.72, 0.73, 0.71, 0.72, 0.655]
    l5 = ["L1", "L2", "L3", "L4", "L1", "L2", "L3", "L4", "L2"]
    show("B5 bridge", d5, l5)
    # B5t bridge exactly in the middle (0.66: 0.07 to both sides) -> tie rule
    show("B5t bridge tie", d5[:-1] + [0.66], l5)
    # B9 support filter on B1b clusters: 2 lines in activity vs 1 line
    one_line = complete_linkage(d1b[:12], ["L1"] * 12, width_fixed(W))
    print("[D] B9 one-line activity clusters:", summary(one_line))
    mixed_d = [0.50, 0.52, 0.80, 1.20, 1.21]
    mixed_l = ["L1", "L2", "L1", "L1", "L3"]
    print("[D] B9 mixed:", summary(complete_linkage(mixed_d, mixed_l, width_fixed(W))))
    # B6 two pipes inside width (limitation)
    d6 = [0.60, 0.61, 0.60, 0.66, 0.67, 0.66]
    l6 = ["L1", "L2", "L3", "L1", "L2", "L3"]
    show("B6 merge-limit", d6, l6)
    # B7 median / n_lines
    show("B7 median", [0.50, 0.52, 0.55, 0.58], ["L1", "L1", "L2", "L3"])
    # B8 split of one pipe -> two supported clusters
    show("B8 split", [0.50, 0.56, 0.62, 0.68], ["L1", "L2", "L3", "L4"])
    # B10 E2 input
    show("B10 E2", [0.50, 0.55, 0.62, 1.00], [1, 2, 3, 1])
    # B3 ties and duplicates
    show("B3 ties", [0.50, 0.55, 0.60, 0.60, 0.65, 0.70, 0.50],
         ["L1", "L2", "L3", "L4", "L1", "L2", "L3"])
    # B11 empty / single
    print("[D] B11 empty", complete_linkage([], [], width_fixed(W)))
    print("[D] B11 single", summary(complete_linkage([0.8], ["L1"], width_fixed(W))))
    # B12 random property check
    viol = 0
    rng12 = np.random.default_rng(0)
    for trial in range(300):
        m = int(rng12.integers(1, 60))
        d = list(np.round(rng12.uniform(0.2, 2.0, m), 3))
        ln = [f"L{int(x)}" for x in rng12.integers(1, 8, m)]
        cl = complete_linkage(d, ln, width_fixed(W))
        p = rng12.permutation(m)
        clp = complete_linkage([d[i] for i in p], [ln[i] for i in p], width_fixed(W))
        if [c[:3] + c[5:] for c in clp] != [c[:3] + c[5:] for c in cl]:
            viol += 1
        for c in cl:
            ds = [x for x, _ in c[5]]
            if max(ds) - min(ds) > W + EDGE_TOL_M:
                viol += 1
        for c in cl:
            if c[4] - c[3] > W + EDGE_TOL_M:
                viol += 1
        # stopping rule: adjacent clusters cannot be merged
        for a, b in zip(cl, cl[1:]):
            if b[4] - a[3] <= W + EDGE_TOL_M:
                viol += 1
        assert sum(c[2] for c in cl) == m
    print("[D] B12 property violations:", viol)


def main() -> None:
    t_start = time.time()
    # eps_r_used uniqueness per activity (metadata only)
    lines = list(csv.DictReader(open(DOC / "fourtu_lines_v1.csv", encoding="utf-8")))
    eps_by_act, n_by_act = {}, {}
    for r in lines:
        eps_by_act.setdefault(r["activity_id"], set()).add(r["eps_r_used"])
        n_by_act[r["activity_id"]] = n_by_act.get(r["activity_id"], 0) + 1
    multi = [a for a, e in eps_by_act.items() if len(e) > 1]
    multi_line = sorted(n for n in n_by_act.values() if n >= 2)
    print(f"[meta] activities {len(eps_by_act)}, with >1 eps_r_used value: {len(multi)}")
    print(f"[meta] activities with >=2 lines: {len(multi_line)}, lines median "
          f"{float(np.median(multi_line))}, p25/p75 {np.quantile(multi_line, [0.25, 0.75])}")

    prototype_cases()

    rq = range_quantiles()
    with open(DOC / "cluster_width_v1_range_q.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rq[0]))
        w.writeheader()
        w.writerows(rq)
    print("[B]", rq)

    mc = noise_mc()
    with open(DOC / "cluster_width_v1_noise_mc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mc[0]))
        w.writeheader()
        w.writerows(mc)
    sd_tau = max(float(r["sd_tau_ns"]) for r in mc if r["sd_tau_ns"] != "")
    sigma_a = max(SIGMA_A_FLOOR_NS, sd_tau)
    print(f"[A] max sd_tau over configs {sd_tau:.4f} ns -> sigma_a used {sigma_a:.4f} ns")

    w_by_n = {r["n"]: r["q95"] for r in rq}
    table = width_table(w_by_n, sigma_a)
    with open(DOC / "cluster_width_v1_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
