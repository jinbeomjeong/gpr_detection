"""Judge M3-1 D1 with the pre-registered rules of
docs/experiments/m31_detection_input_tiling.md v0.3.3 (commit 9fb5aab) §8-2 (3)(4).

Reads runs/detect/<prefix><k>/eval/m31_d1/metrics.json for k = 0..4 (written by
m31_d1_shift_scale.py), recomputes sha256 of the weights and eval lists, and
writes a JSON verdict (--out). CPU only, read-only on inputs.

Aggregation (v0.3.3 §8-2 (3)(a)-(e)):
  - units %p (fractions x 100);
  - every paired difference, min/max and "worse of two series" per fold first,
    then the unweighted 5-fold mean is compared with the 3.0 %p threshold;
  - baseline check on 15 values (5 folds x hyperbola/cavity/manhole),
    |AP50 - M1-10 AP50| <= 1.0 %p each, any failure -> no verdict;
  - rule O clause order: both fail -> 50 % -> 75 %;
  - compression loss (report only): recall at g = 0.79 and 0.53 by linear
    interpolation in log g between neighbouring S points, per fold, then mean.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "runs" / "detect"
PREFIX = "yolo11m_aihubv1_20261101_fold"
FOLDS = (0, 1, 2, 3, 4)
THRESHOLD_PP = 3.0
BASELINE_TOL_PP = 1.0
BASELINE_CLASSES = ("pipe_hyperbola", "cavity", "manhole")
EXPECTED_HYP_NGT = (71, 45, 98, 147, 142)
EXPECTED_WEIGHTS_SHA8 = ("5300f34b", "cf9423f1", "8dfa5713", "43ae65ee", "19602860")
HYP = "pipe_hyperbola"
S_POINTS = (("g0.50_dx+0", 0.50), ("g0.74_dx+0", 0.74), ("g0.87_dx+0", 0.87),
            ("g1.00_dx+0", 1.00), ("g1.13_dx+0", 1.13), ("g1.28_dx+0", 1.28),
            ("gp6_dx+0", 6.0 / 4.06), ("g1.67_dx+0", 1.67))
JUDGED = ["g0.50_dx+0", "g0.74_dx+0", "g1.00_dx+0", "gp6_dx+0", "g1.67_dx+0",
          "g1.00_dx-32", "g1.00_dx+32", "g1.00_dx-64", "g1.00_dx+64",
          "gp6_dx-32", "gp6_dx+32", "g1.00_dx+0_fillL64", "g1.00_dx+0_fillR64"]
COMPRESSION_G = (0.79, 0.53)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pp(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


def rec(cond: dict, name: str, cls: str = HYP) -> float:
    return pp(cond[name]["classes"][cls]["recall_conf025"])


def interp_log_g(values: dict[float, float], g: float) -> tuple[float, float, float]:
    """Linear interpolation in log g; returns (value, nearest_g, nearest_value)."""
    gs = sorted(values)
    lo = max(x for x in gs if x <= g)
    hi = min(x for x in gs if x >= g)
    if lo == hi:
        v = values[lo]
    else:
        t = (math.log(g) - math.log(lo)) / (math.log(hi) - math.log(lo))
        v = values[lo] + t * (values[hi] - values[lo])
    nearest = min(gs, key=lambda x: abs(math.log(x) - math.log(g)))
    return v, nearest, values[nearest]


def mean_sd(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=float)
    return {"per_fold": [round(float(x), 3) for x in arr], "mean": round(float(arr.mean()), 3),
            "sd": round(float(arr.std(ddof=1)), 3) if arr.size > 1 else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subdir", default="m31_d1")
    args = parser.parse_args(argv)

    folds = {}
    identity = []
    for k in FOLDS:
        run = RUN_ROOT / f"{PREFIX}{k}"
        m = json.loads((run / "eval" / args.subdir / "metrics.json").read_text(encoding="utf-8"))
        folds[k] = m
        w = Path(m["weights"])
        e = Path(m["eval_list"])
        m110 = json.loads((run / "eval" / "last_crop" / "metrics.json").read_text(encoding="utf-8"))
        w_sha = sha256(w)
        e_sha = sha256(e)
        identity.append({
            "fold": k, "weights": str(w), "weights_sha256": w_sha,
            "weights_sha8_expected": EXPECTED_WEIGHTS_SHA8[k],
            "weights_match_audit": w_sha.startswith(EXPECTED_WEIGHTS_SHA8[k]),
            "weights_match_m110_metrics": w_sha == m110["weights"]["sha256"],
            "eval_list": str(e), "eval_list_sha256": e_sha,
            "eval_list_match_m110_metrics": e_sha == m110["eval_images"]["sha256"],
            "n_images": m["n_images"], "n_images_m110": m110["eval_images"]["count"],
        })

    # 1. baseline check (15 values)
    baseline = []
    for k, m in folds.items():
        c = m["conditions"]["g1.00_dx+0"]["classes"]
        for cls in BASELINE_CLASSES:
            d = pp(c[cls]["ap50"]) - pp(m["m110_ap50"][cls])
            baseline.append({"fold": k, "class": cls, "ap50_d1": round(pp(c[cls]["ap50"]), 3),
                             "ap50_m110": round(pp(m["m110_ap50"][cls]), 3), "diff_pp": round(d, 3),
                             "pass": abs(d) <= BASELINE_TOL_PP})
    baseline_pass = all(b["pass"] for b in baseline)

    # 4. n_gt checks
    ngt = []
    for k, m in folds.items():
        base = m["conditions"]["g1.00_dx+0"]["classes"]
        row = {"fold": k, "hyp_ngt_g1dx0": base[HYP]["n_gt"], "expected": EXPECTED_HYP_NGT[k],
               "match": base[HYP]["n_gt"] == EXPECTED_HYP_NGT[k], "judged_all_kept": True,
               "ignored_by_condition": {}}
        for name, cond in m["conditions"].items():
            ign = {cls: v["n_ignored"] for cls, v in cond["classes"].items() if v["n_ignored"]}
            if ign:
                row["ignored_by_condition"][name] = ign
            if name in JUDGED:
                # hyperbola GT must all stay; other classes reported
                if cond["classes"][HYP]["n_ignored"] != 0 or cond["classes"][HYP]["n_gt"] != base[HYP]["n_gt"]:
                    row["judged_all_kept"] = False
        ngt.append(row)

    out = {"schema": "m31_d1_verdict/1", "rules": "m31_detection_input_tiling.md v0.3.3 §8-2 (3)(4)",
           "identity": identity, "baseline_check": {"pass": baseline_pass, "values": baseline},
           "n_gt": ngt}
    if not baseline_pass:
        out["verdict"] = "no verdict: baseline check failed"
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"baseline_pass": False}))
        return 1

    # per-fold quantities (pp)
    q: dict[str, list[float]] = {k: [] for k in (
        "R_g1", "R_g1478", "R_g167", "dC_best", "dC_worst", "dB",
        "X1_d32", "X1_d64", "X1_d96", "X6_d32", "X6_d64", "dF",
        "worse_d32", "worse_d64", "E406", "E6",
        "R_g079_interp", "R_g053_interp", "R_g079_nearest", "R_g053_nearest")}
    s_curve = {name: [] for name, _ in S_POINTS}
    secondary = {cls: {"dC_worst": [], "X1_d32": [], "X1_d64": [], "dF": []} for cls in ("cavity", "manhole")}
    map50 = {name: [] for name in ("g1.00_dx+0", "gp6_dx+0", "g1.67_dx+0", "g0.50_dx+0", "g0.74_dx+0")}
    for k, m in folds.items():
        c = m["conditions"]
        r = {name: rec(c, name) for name in c}
        for name, _ in S_POINTS:
            s_curve[name].append(r[name])
        r1, r6, r167 = r["g1.00_dx+0"], r["gp6_dx+0"], r["g1.67_dx+0"]
        q["R_g1"].append(r1); q["R_g1478"].append(r6); q["R_g167"].append(r167)
        q["dC_best"].append(r1 - max(r6, r167))
        q["dC_worst"].append(r1 - min(r6, r167))
        q["dB"].append(r["g0.74_dx+0"] - r["g0.50_dx+0"])
        x1 = {d: r1 - 0.5 * (r[f"g1.00_dx+{d}"] + r[f"g1.00_dx-{d}"]) for d in (32, 64, 96)}
        x6 = {d: r6 - 0.5 * (r[f"gp6_dx+{d}"] + r[f"gp6_dx-{d}"]) for d in (32, 64)}
        dF = r1 - 0.5 * (r["g1.00_dx+0_fillL64"] + r["g1.00_dx+0_fillR64"])
        q["X1_d32"].append(x1[32]); q["X1_d64"].append(x1[64]); q["X1_d96"].append(x1[96])
        q["X6_d32"].append(x6[32]); q["X6_d64"].append(x6[64]); q["dF"].append(dF)
        q["worse_d32"].append(max(x1[32], x6[32])); q["worse_d64"].append(max(x1[64], x6[64]))
        q["E406"].append(np.mean([r["g0.50_dx+0"], r["g0.87_dx+0"], r["g1.13_dx+0"]]))
        q["E6"].append(np.mean([r["g0.74_dx+0"], r["g1.28_dx+0"], r["g1.67_dx+0"]]))
        sv = {g: r[name] for name, g in S_POINTS}
        for g, key in ((0.79, "079"), (0.53, "053")):
            v, ng, nv = interp_log_g(sv, g)
            q[f"R_g{key}_interp"].append(v); q[f"R_g{key}_nearest"].append(nv)
        for cls in secondary:
            rc = {name: rec(c, name, cls) for name in c}
            secondary[cls]["dC_worst"].append(rc["g1.00_dx+0"] - min(rc["gp6_dx+0"], rc["g1.67_dx+0"]))
            secondary[cls]["X1_d32"].append(rc["g1.00_dx+0"] - 0.5 * (rc["g1.00_dx+32"] + rc["g1.00_dx-32"]))
            secondary[cls]["X1_d64"].append(rc["g1.00_dx+0"] - 0.5 * (rc["g1.00_dx+64"] + rc["g1.00_dx-64"]))
            secondary[cls]["dF"].append(rc["g1.00_dx+0"] - 0.5 * (rc["g1.00_dx+0_fillL64"] + rc["g1.00_dx+0_fillR64"]))
        for name in map50:
            map50[name].append(pp(c[name]["map50"]))

    agg = {k: mean_sd(v) for k, v in q.items()}
    T = THRESHOLD_PP
    # Rule P
    dCw, dB = agg["dC_worst"]["mean"], agg["dB"]["mean"]
    p_star = 6.0 if (dCw <= T and dB >= T) else 4.06
    rule_p = {"dC_worst_mean_pp": dCw, "dB_mean_pp": dB, "p_star": p_star,
              "small_hyperbola_followup": p_star == 4.06 and dB >= T}
    # Rule O
    if p_star == 4.06:
        series = "X1"
        dF_mean = agg["dF"]["mean"]
        contaminated = dF_mean > T
        if contaminated:
            d32, d64 = agg["worse_d32"]["mean"], agg["worse_d64"]["mean"]
        else:
            d32, d64 = agg["X1_d32"]["mean"], agg["X1_d64"]["mean"]
    else:
        series, dF_mean, contaminated = "X6", None, False
        d32, d64 = agg["X6_d32"]["mean"], agg["X6_d64"]["mean"]
    if d32 > T:
        overlap = "both_fail (provisional 75 %)"
    elif d64 <= T:
        overlap = "50 % (O 128)"
    else:
        overlap = "75 % (O 192)"
    rule_o = {"series": series, "dF_mean_pp": dF_mean, "fill_contaminated": contaminated,
              "delta32_used_pp": d32, "delta64_used_pp": d64, "overlap": overlap}

    out.update({
        "per_fold_and_mean_pp": agg,
        "hyperbola_recall_S_curve_pp": {name: mean_sd(v) for name, v in s_curve.items()},
        "secondary_classes_pp": {cls: {k: mean_sd(v) for k, v in d.items()} for cls, d in secondary.items()},
        "map50_pp": {k: mean_sd(v) for k, v in map50.items()},
        "rule_p": rule_p, "rule_o": rule_o,
        "verdict": {"p_star": p_star, "overlap": overlap},
    })
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline_pass": True, "p_star": p_star, "overlap": overlap,
                      "dC_worst": dCw, "dB": dB, "d32": d32, "d64": d64, "dF": dF_mean}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
