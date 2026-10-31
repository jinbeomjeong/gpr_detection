"""Aggregate the M1-7b pipe_band merge v2 vs v1 comparison and apply the pre-registered decision rule.

Rule: docs/experiments/band_merge_v2_compare.md section 6-1 (fixed before the 2026-10-31 run, applied verbatim).
Inputs: runs/detect/band_v2_eval/<run>__gt_<v1|v2>/eval/last_<crop|image>/metrics.json (eval_detector, gpr_eval/1).
Decision basis: last.pt, crop unit. Image unit is computed with the same formula for reference only.

Usage: python -I analysis/detection/band_v2_compare_aggregate.py [--root C:/workspace/gpr_detection] [--out <json>]
Writes JSON (default analysis/detection/band_v2_compare_aggregate.json); prints markdown tables and the
decision as a one-line JSON on the last stdout line. Exit 0 on success, 2 on missing input, 3 on internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNS = {
    ("v1", 0): "m17_aug0",
    ("v1", 1): "m17_aug0_s1",
    ("v2", 0): "yolo11m_bandv2_aug0_20261031_fold0_s0",
    ("v2", 1): "yolo11m_bandv2_aug0_20261031_fold0_s1",
}
GTS = ["v2", "v1"]
UNITS = ["crop", "image"]
CLASSES = ["pipe_hyperbola", "pipe_band", "cavity", "manhole"]
# M1-7 v1 seed difference (crop last.pt, v1 GT, M1-7 section 8-4); floor for tau under both GTs (%p).
TAU_FLOOR = {"band": 2.29, "pipe": 4.26, "map": 1.97}
METRICS = ["band", "pipe", "map"]


def load(root: Path, run: str, gt: str, unit: str) -> dict:
    p = root / "runs" / "detect" / "band_v2_eval" / f"{run}__gt_{gt}" / "eval" / f"last_{unit}" / "metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["fold"] == "0", p
    assert m["unit"] == unit, p
    assert m["weights"]["name"] == "last", p
    ap = {c["name"]: c["ap50"] * 100 for c in m["per_class"]}
    ap5095 = {c["name"]: c["ap50_95"] * 100 for c in m["per_class"]}
    pr = {c["name"]: (c["precision"], c["recall"]) for c in m["per_class"]}
    inst = {c["name"]: c["instances"] for c in m["per_class"]}
    cm = m["confusion_matrix_at_confusion_conf"]
    assert abs(cm["conf_threshold"] - 0.25) < 1e-9, p
    labels, mat = cm["labels"], cm["matrix"]
    bg, band = labels.index("background"), labels.index("pipe_band")
    return {
        "path": str(p),
        "count": m["eval_images"]["count"],
        "eval_images_sha256": m["eval_images"]["sha256"],
        "weights_sha256": m["weights"]["sha256"],
        "ap50": ap,
        "ap50_95": ap5095,
        "precision_recall": pr,
        "instances": inst,
        "band": ap["pipe_band"],
        "pipe": m["pipe_ap50_mean"]["value"] * 100,
        "map": m["map50"] * 100,
        "map50_95": m["map50_95"] * 100,
        "cm025": {
            "band_tp": mat[band][band],
            "bg_fp_band": mat[band][bg],          # predicted band, no GT (background column)
            "band_fn": mat[bg][band],             # GT band missed
            "band_pred_total": sum(mat[band]),    # all band predictions at conf 0.25
            "band_gt_total": sum(mat[i][band] for i in range(len(labels))),
        },
    }


def mean(a: float, b: float) -> float:
    return (a + b) / 2.0


def evaluate_gt(data: dict, gt: str, unit: str) -> dict:
    out = {"values": {}, "delta": {}, "tau": {}, "tau_parts": {}}
    for k in METRICS:
        v1 = [data[("v1", s, gt, unit)][k] for s in (0, 1)]
        v2 = [data[("v2", s, gt, unit)][k] for s in (0, 1)]
        d = mean(*v2) - mean(*v1)
        parts = {"m17_floor": TAU_FLOOR[k], "v1_seed_diff": abs(v1[0] - v1[1]), "v2_seed_diff": abs(v2[0] - v2[1])}
        out["values"][k] = {"v1_s0": v1[0], "v1_s1": v1[1], "v2_s0": v2[0], "v2_s1": v2[1]}
        out["delta"][k] = d
        out["tau"][k] = max(parts.values())
        out["tau_parts"][k] = parts
    return out


def classify(g2: dict, g1: dict) -> dict:
    d, t = g2["delta"], g2["tau"]
    dp, tp = g1["delta"], g1["tau"]
    v = g2["values"]["band"]
    nonoverlap = min(v["v2_s0"], v["v2_s1"]) > max(v["v1_s0"], v["v1_s1"])
    u_parts = {
        "band_gt_tau": d["band"] > t["band"],
        "pipe_gt_tau": d["pipe"] > t["pipe"],
        "band_range_nonoverlap": nonoverlap,
        "map_ge_minus_tau": d["map"] >= -t["map"],
    }
    n_parts = {
        "band_ge_minus_tau": dp["band"] >= -tp["band"],
        "pipe_ge_minus_tau": dp["pipe"] >= -tp["pipe"],
        "map_ge_minus_tau": dp["map"] >= -tp["map"],
    }
    strict_alt = {k: dp[k] > tp[k] for k in METRICS}  # not adopted; reported only
    U = all(u_parts.values())
    N = all(n_parts.values())
    band_up = d["band"] > t["band"]
    if d["band"] < -t["band"] or d["map"] < -t["map"] or (d["band"] <= t["band"] and not N):
        cls, order = "열세", 1
    elif U and N:
        cls, order = "통과", 2
    elif band_up and not N:
        cls, order = "부분 통과-규약 교환 의심", 3
    elif band_up and N and not (u_parts["pipe_gt_tau"] and u_parts["band_range_nonoverlap"]):
        cls, order = "부분 통과-약함", 4
    else:
        cls, order = "동등", 5
    return {"class": cls, "order": order, "U": U, "N": N, "U_parts": u_parts, "N_parts": n_parts,
            "strict_alt_v1gt_delta_gt_tau": strict_alt}


def fmt_table(gres: dict, gt: str, cond: dict, cond_name: str) -> str:
    names = {"band": "pipe_band AP50", "pipe": "관로 평균 AP50", "map": "전체 mAP50"}
    lines = [f"| 지표 ({gt} 정답) | v1 s0 | v1 s1 | v2 s0 | v2 s1 | Δ | τ | {cond_name} |", "|---|---|---|---|---|---|---|---|"]
    for k in METRICS:
        v = gres["values"][k]
        lines.append(f"| {names[k]} | {v['v1_s0']:.2f} | {v['v1_s1']:.2f} | {v['v2_s0']:.2f} | {v['v2_s1']:.2f} | "
                     f"{gres['delta'][k]:+.2f} | {gres['tau'][k]:.2f} | {cond.get(k, '')} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/workspace/gpr_detection")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out_path = Path(a.out) if a.out else root / "analysis" / "detection" / "band_v2_compare_aggregate.json"

    data = {}
    try:
        for (lab, s), run in RUNS.items():
            for gt in GTS:
                for unit in UNITS:
                    data[(lab, s, gt, unit)] = load(root, run, gt, unit)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"missing input: {e.filename}"}))
        return 2

    # Sanity: same eval image list per (gt, unit); weights identical across GT views of one run.
    checks = {}
    for gt in GTS:
        for unit in UNITS:
            shas = {data[(l, s, gt, unit)]["eval_images_sha256"] for (l, s) in RUNS}
            checks[f"eval_images_same_{gt}_{unit}"] = len(shas) == 1
    for (l, s) in RUNS:
        w = {data[(l, s, gt, u)]["weights_sha256"] for gt in GTS for u in UNITS}
        checks[f"weights_same_across_views_{l}_s{s}"] = len(w) == 1

    # Reproducibility: v1 runs under v1 GT vs original M1-7 eval folders.
    repro = {}
    for s in (0, 1):
        for unit in UNITS:
            p = root / "runs" / "detect" / RUNS[("v1", s)] / "eval" / f"last_{unit}" / "metrics.json"
            if p.exists():
                m = json.loads(p.read_text(encoding="utf-8"))
                new = data[("v1", s, "v1", unit)]
                repro[f"v1_s{s}_{unit}"] = {
                    "orig_map": m["map50"] * 100, "view_map": new["map"],
                    "orig_band": {c["name"]: c["ap50"] for c in m["per_class"]}["pipe_band"] * 100,
                    "view_band": new["band"],
                    "max_abs_diff_pp": max(abs(m["map50"] * 100 - new["map"]),
                                           abs({c["name"]: c["ap50"] for c in m["per_class"]}["pipe_band"] * 100 - new["band"]),
                                           abs(m["pipe_ap50_mean"]["value"] * 100 - new["pipe"])),
                }

    result = {"schema": "band_v2_compare/1", "rule": "docs/experiments/band_merge_v2_compare.md#6-1",
              "tau_floor_m17": TAU_FLOOR, "checks": checks, "repro_v1_vs_m17": repro, "units": {}}
    for unit in UNITS:
        g2 = evaluate_gt(data, "v2", unit)
        g1 = evaluate_gt(data, "v1", unit)
        dec = classify(g2, g1)
        per_class = {}
        for gt in GTS:
            per_class[gt] = {}
            for c in CLASSES:
                vals = {f"{l}_s{s}": data[(l, s, gt, unit)]["ap50"][c] for (l, s) in RUNS}
                vals["delta"] = mean(vals["v2_s0"], vals["v2_s1"]) - mean(vals["v1_s0"], vals["v1_s1"])
                vals["instances"] = data[("v1", 0, gt, unit)]["instances"][c]
                per_class[gt][c] = vals
            per_class[gt]["map50_95"] = {f"{l}_s{s}": data[(l, s, gt, unit)]["map50_95"] for (l, s) in RUNS}
        band_detail = {gt: {f"{l}_s{s}": {"precision": data[(l, s, gt, unit)]["precision_recall"]["pipe_band"][0],
                                          "recall": data[(l, s, gt, unit)]["precision_recall"]["pipe_band"][1],
                                          "ap50_95": data[(l, s, gt, unit)]["ap50_95"]["pipe_band"],
                                          **data[(l, s, gt, unit)]["cm025"]}
                            for (l, s) in RUNS} for gt in GTS}
        result["units"][unit] = {
            "role": "decision" if unit == "crop" else "reference",
            "images": data[("v1", 0, "v2", unit)]["count"],
            "band_gt": {gt: data[("v1", 0, gt, unit)]["instances"]["pipe_band"] for gt in GTS},
            "gt_v2": g2, "gt_v1": g1, "decision": dec, "per_class_ap50": per_class, "band_detail": band_detail,
        }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for unit in UNITS:
        r = result["units"][unit]
        dec = r["decision"]
        u = dec["U_parts"]
        n = dec["N_parts"]
        print(f"## {unit} ({r['role']}), images {r['images']}, band GT {r['band_gt']}")
        print(fmt_table(r["gt_v2"], "v2", {"band": f"Δ>τ {u['band_gt_tau']}, 비겹침 {u['band_range_nonoverlap']}",
                                            "pipe": f"Δ>τ {u['pipe_gt_tau']}", "map": f"Δ≥−τ {u['map_ge_minus_tau']}"}, "U 조건"))
        print()
        print(fmt_table(r["gt_v1"], "v1", {k: f"Δ'≥−τ' {n[k + '_ge_minus_tau']}" for k in METRICS}, "N 조건"))
        print(f"U={dec['U']} N={dec['N']} -> {dec['order']} {dec['class']}")
        print()
    print("checks:", json.dumps(checks), "repro:", json.dumps({k: round(v["max_abs_diff_pp"], 4) for k, v in repro.items()}))
    c = result["units"]["crop"]
    final = {
        "decision": c["decision"]["class"], "order": c["decision"]["order"], "U": c["decision"]["U"], "N": c["decision"]["N"],
        "basis": "crop/last.pt/fold0",
        "gt_v2": {k: {"delta": round(c["gt_v2"]["delta"][k], 2), "tau": round(c["gt_v2"]["tau"][k], 2)} for k in METRICS},
        "gt_v1": {k: {"delta": round(c["gt_v1"]["delta"][k], 2), "tau": round(c["gt_v1"]["tau"][k], 2)} for k in METRICS},
        "image_reference": result["units"]["image"]["decision"]["class"],
        "out": str(out_path),
    }
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    # -I ignores PYTHONIOENCODING; force UTF-8 so Korean/minus signs print on a cp949 console.
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"internal: {type(e).__name__}: {e}"}))
        sys.exit(3)
