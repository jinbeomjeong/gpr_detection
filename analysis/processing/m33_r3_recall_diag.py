"""M3-3 pre-close item 2: why is the R3 pseudo-band recall so low?

Read-only re-aggregation of the published R3 results (P4/P5 judgment is out,
2026-11-16) and the 2026-11-20 change report. No detector, no injection, no
writes outside ``analysis/processing/``.

Inputs (read only)
- runs/eval/m33/r3/injected/{truths.csv, fold*/inj_matches.csv, fold*/inj_boxes.csv}
- runs/eval/m33/r3/uninjected_ids/{lines.csv, fold*/boxes.csv}
- runs/eval/m33/r2/{lines.csv, fold*/boxes.csv}          (4TU un-injected boxes)
- runs/eval/m33/r3_change_report/{realizations.csv, lines.csv}

Outputs
- analysis/processing/m33_r3_recall_diag.json   (tables)
- analysis/processing/m33_r3_recall_diag_units.csv (truth x fold rows)

Run: conda run -n gpr-dev python -I -B analysis/processing/m33_r3_recall_diag.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "runs" / "eval" / "m33" / "r3"
R2 = ROOT / "runs" / "eval" / "m33" / "r2"
CR = ROOT / "runs" / "eval" / "m33" / "r3_change_report"
OUT_JSON = ROOT / "analysis" / "processing" / "m33_r3_recall_diag.json"
OUT_CSV = ROOT / "analysis" / "processing" / "m33_r3_recall_diag_units.csv"

FOLDS = (0, 1, 2, 3, 4)
CONF = 0.25
CLASSES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
SNRS = (0.0, 6.0, 12.0, 18.0)


def lower_priority() -> None:
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)  # BelowNormal
    except Exception:  # pragma: no cover - non-Windows
        pass


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tf(value: str) -> bool:
    return str(value).strip().lower() == "true"


# ----------------------------------------------------------------- loading


def load_boxes(path: Path, keep) -> list[dict]:
    rows = []
    for row in read(path):
        if row["pp"] != "pre" or not tf(row["in_grid"]):
            continue
        if not keep(row):
            continue
        score = float(row["score"])
        if score <= CONF:
            continue
        rows.append(
            {
                "fold": int(row["fold"]),
                "line_id": row["line_id"],
                "realization_id": row.get("realization_id", ""),
                "cls": row["class_name"],
                "score": score,
                "i0": int(row["i0"]),
                "i1": int(row["i1"]),
                "k0": int(row["k0"]),
                "k1": int(row["k1"]),
            }
        )
    return rows


def shared_rows(a: dict, b: dict) -> int:
    return min(a["k1"], b["k1"]) - max(a["k0"], b["k0"]) + 1


def overlaps_traces(box: dict, i0: int, i1: int) -> bool:
    return box["i0"] <= i1 and box["i1"] >= i0


def covered(truth: dict, boxes) -> int:
    pieces = sorted(
        (max(b["i0"], truth["i0"]), min(b["i1"], truth["i1"]))
        for b in boxes
        if overlaps_traces(b, truth["i0"], truth["i1"])
    )
    total, reach = 0, truth["i0"] - 1
    for start, stop in pieces:
        start = max(start, reach + 1)
        if stop >= start:
            total += stop - start + 1
            reach = stop
    return total


def iou(a: dict, b: dict) -> float:
    w = min(a["i1"], b["i1"]) - max(a["i0"], b["i0"]) + 1
    h = shared_rows(a, b)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = lambda r: (r["i1"] - r["i0"] + 1) * (r["k1"] - r["k0"] + 1)
    return inter / (area(a) + area(b) - inter)


# ------------------------------------------------------------- aggregation


def pooled(units, key_cov: str) -> float | None:
    """Length-weighted recall over all units (simple pooling)."""
    num = sum(u[key_cov] for u in units)
    den = sum(u["L"] for u in units)
    return None if den == 0 else num / den


def p4_style(units, key_cov: str) -> float | None:
    """P4 'pooled' definition: SNR x polarity cells equal, length-weighted in
    the cell, mean over folds."""
    per_fold = []
    for fold in FOLDS:
        cells = defaultdict(lambda: [0, 0])
        for u in units:
            if u["fold"] != fold:
                continue
            c = cells[(u["snr"], u["pol"])]
            c[0] += u[key_cov]
            c[1] += u["L"]
        vals = [n / d for n, d in cells.values() if d > 0]
        if vals:
            per_fold.append(sum(vals) / len(vals))
    return None if not per_fold else sum(per_fold) / len(per_fold)


COV_KEYS = (
    "cov_band50",
    "cov_any50",
    "cov_hyp50",
    "cov_cav50",
    "cov_man50",
    "cov_band_anyrow",
    "cov_band_boxfrac",
    "cov_any_anyrow",
)


def table(units, key_fn, keys=COV_KEYS) -> dict:
    groups = defaultdict(list)
    for u in units:
        k = key_fn(u)
        if k is None:
            continue
        groups[k].append(u)
    out = {}
    for k in sorted(groups, key=lambda x: (str(type(x)), x)):
        g = groups[k]
        row = {
            "n_units": len(g),
            "n_positions": len({u["position_id"] for u in g}),
            "n_lines": len({u["line_id"] for u in g}),
        }
        for key in keys:
            v = pooled(g, key)
            row[key] = None if v is None else round(v, 4)
        out[str(k)] = row
    return out


def quantiles(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    v = sorted(x for x in values if x is not None and not math.isnan(x))
    if not v:
        return {"n": 0}
    out = {"n": len(v)}
    for q in qs:
        pos = q * (len(v) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        out[f"p{int(q * 100)}"] = round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 4)
    return out


def bin_of(value: float | None, edges, labels):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


# -------------------------------------------------------------------- main


def main() -> None:
    lower_priority()
    truths = read(R3 / "injected" / "truths.csv")
    matches = {}
    for fold in FOLDS:
        for row in read(R3 / "injected" / f"fold{fold}" / "inj_matches.csv"):
            matches[(row["injection_id"], fold)] = row

    inj_boxes = defaultdict(list)
    for fold in FOLDS:
        for b in load_boxes(R3 / "injected" / f"fold{fold}" / "inj_boxes.csv", lambda r: True):
            inj_boxes[(b["realization_id"], fold)].append(b)

    uninj = defaultdict(list)
    for fold in FOLDS:
        for b in load_boxes(R3 / "uninjected_ids" / f"fold{fold}" / "boxes.csv", lambda r: True):
            uninj[(b["line_id"], fold)].append(b)
        for b in load_boxes(
            R2 / f"fold{fold}" / "boxes.csv",
            lambda r: r.get("recipe") == "r0" and r.get("orientation") == "original",
        ):
            uninj[(b["line_id"], fold)].append(b)

    real = {r["realization_id"]: r for r in read(CR / "realizations.csv")}
    lines = {r["line_id"]: r for r in read(CR / "lines.csv")}
    # L1 truth box rows at the background (plan) scale; realizations whose
    # period estimate moved are drawn at a slightly different vertical scale.
    positions = {r["position_id"]: r for r in read(R3 / "plan" / "positions.csv")}

    units = []
    for t in truths:
        truth = {k: int(t[k]) for k in ("i0", "i1", "k0", "k1")}
        h = truth["k1"] - truth["k0"] + 1
        L = truth["i1"] - truth["i0"] + 1
        line = lines[t["line_id"]]
        period = float(line["period_bg_samples"])
        kz = float(line["k_zero_bg"])
        n_tr = int(line["n_traces"])
        rr = real[t["realization_id"]]
        dp_rel = float(rr["period_dp_rel"])
        p_inj = float(rr["period_inj_samples"])
        apex = float(t["apex_sample"])
        surv = t["survival_ratio"]
        surv = float(surv) if surv not in ("", None) else None
        for fold in FOLDS:
            m = matches[(t["injection_id"], fold)]
            boxes = inj_boxes.get((t["realization_id"], fold), [])
            by_cls = defaultdict(list)
            for b in boxes:
                by_cls[b["cls"]].append(b)
            half = 0.5 * h

            def c50(bs):
                return covered(truth, [b for b in bs if shared_rows(truth, b) >= half])

            band = by_cls["pipe_band"]
            cov_band50 = c50(band)
            cov_any50 = c50(boxes)
            assert cov_band50 == int(m["covered_pre"]), (t["injection_id"], fold)
            assert cov_any50 == int(m["covered_any_class_pre"]), (t["injection_id"], fold)
            cov_band_anyrow = covered(truth, [b for b in band if shared_rows(truth, b) >= 1])
            cov_band_boxfrac = covered(
                truth,
                [
                    b
                    for b in band
                    if shared_rows(truth, b) >= 0.5 * min(h, b["k1"] - b["k0"] + 1)
                ],
            )
            cov_any_anyrow = covered(truth, [b for b in boxes if shared_rows(truth, b) >= 1])
            # boxes touching the truth rectangle: are they new (no IoU>=0.5 same-class
            # pair in the un-injected line/fold)?
            touching = [
                b
                for b in boxes
                if shared_rows(truth, b) >= 1 and overlaps_traces(b, truth["i0"], truth["i1"])
            ]
            pool = uninj.get((t["line_id"], fold), [])
            n_new = sum(
                1
                for b in touching
                if not any(o["cls"] == b["cls"] and iou(o, b) >= 0.5 for o in pool)
            )
            # geometry of pipe_band boxes that pass the 50 % rule
            matched = [b for b in band if shared_rows(truth, b) >= half and overlaps_traces(b, truth["i0"], truth["i1"])]
            units.append(
                {
                    "injection_id": t["injection_id"],
                    "realization_id": t["realization_id"],
                    "position_id": t["position_id"],
                    "line_id": t["line_id"],
                    "dataset": t["dataset"],
                    "freq": t["ids_freq_stratum"],
                    "model_group": t["model_group"],
                    "response_id": t["response_id"],
                    "main_cell": tf(t["main_cell"]),
                    "lw": t["lw_stratum"],
                    "px": t["px_stratum"],
                    "L": L,
                    "L_nominal": int(t["band_length_traces"]),
                    "L_periods_px": float(t["length_px"]) / 6.0,
                    "w_eff": int(t["w_eff"]),
                    "short_line": int(t["w_eff"]) < 301,
                    "n_traces": n_tr,
                    "snr": float(t["snr_db"]),
                    "pol": int(t["polarity"]),
                    "fold": fold,
                    "h_P": h / period,
                    "truth_yc_px": 0.5
                    * (
                        float(positions[t["position_id"]]["truth_y1_px"])
                        + float(positions[t["position_id"]]["truth_y2_px"])
                    ),
                    "truth_y1_px": float(positions[t["position_id"]]["truth_y1_px"]),
                    "apex_depth_P": (apex - kz) / period,
                    "survival": surv,
                    "dp_rel": dp_rel,
                    "period_changed": abs(p_inj - period) > 0.05 * period,
                    "in_hzone": tf(t["in_horizontal_box_zone"]),
                    "n_clipped": int(t["n_clipped_cells"]),
                    "cov_band50": cov_band50,
                    "cov_any50": cov_any50,
                    "cov_hyp50": c50(by_cls["pipe_hyperbola"]),
                    "cov_cav50": c50(by_cls["cavity"]),
                    "cov_man50": c50(by_cls["manhole"]),
                    "cov_band_anyrow": cov_band_anyrow,
                    "cov_band_boxfrac": cov_band_boxfrac,
                    "cov_any_anyrow": cov_any_anyrow,
                    "n_touching": len(touching),
                    "n_touching_new": n_new,
                    "matched_h_P": [(b["k1"] - b["k0"] + 1) / period for b in matched],
                    "matched_dc_P": [
                        ((b["k0"] + b["k1"]) / 2 - (truth["k0"] + truth["k1"]) / 2) / period
                        for b in matched
                    ],
                    "matched_w_over_L": [(b["i1"] - b["i0"] + 1) / L for b in matched],
                    "matched_score": [b["score"] for b in matched],
                }
            )

    def p4_scope(u):
        return (
            u["dataset"] == "gz_ids"
            and u["freq"] == "main_400"
            and u["model_group"] == "A0"
            and u["main_cell"]
            and u["lw"] in ("lt033", "033_05")
        )

    scopes = {
        "p4": [u for u in units if p4_scope(u)],
        "ids_main400_all_lw": [
            u for u in units if u["dataset"] == "gz_ids" and u["freq"] == "main_400" and u["main_cell"]
        ],
        "4tu_A0_main": [u for u in units if u["dataset"] == "4tu" and u["model_group"] == "A0" and u["main_cell"]],
        "4tu_A1_main": [u for u in units if u["dataset"] == "4tu" and u["model_group"] == "A1" and u["main_cell"]],
    }

    out = {"inputs": {}, "scopes": {}}
    for path in [
        R3 / "injected" / "truths.csv",
        CR / "realizations.csv",
        CR / "lines.csv",
        R3 / "uninjected_ids" / "lines.csv",
        R3 / "plan" / "positions.csv",
    ]:
        import hashlib

        out["inputs"][str(path.relative_to(ROOT)).replace(os.sep, "/")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    sv_edges = (0.2, 0.4, 0.6, 0.8, 1e9)
    sv_labels = ("s<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8")
    dep_edges = (4, 6, 8, 12, 1e9)
    dep_labels = ("<4P", "4-6P", "6-8P", "8-12P", ">=12P")
    dp_edges = (-0.05, 0.05, 1e9)
    dp_labels = ("dP<-5%", "|dP|<=5%", "dP>+5%")
    lp_edges = (12.4, 1e9)
    lp_labels = ("L<12.4P(AIHub p5)", "L>=12.4P")
    # AIHub lpipe label ymin (256 px crops): min 36.9, p5 46, p50 57, p95 79
    # (docs/processing/detection_image_scale_survey_v1.json "lpipe")
    y_edges = (37, 46, 57, 79, 1e9)
    y_labels = ("y1<37", "37-46", "46-57", "57-79", ">=79")

    for name, us in scopes.items():
        s = {
            "n_units": len(us),
            "n_lines": len({u["line_id"] for u in us}),
            "n_positions": len({u["position_id"] for u in us}),
            "pooled": {k: (None if pooled(us, k) is None else round(pooled(us, k), 4)) for k in COV_KEYS},
            "p4_style": {k: (None if p4_style(us, k) is None else round(p4_style(us, k), 5)) for k in ("cov_band50", "cov_any50", "cov_band_anyrow", "cov_any_anyrow")},
            "by_snr": table(us, lambda u: u["snr"]),
            "by_pol": table(us, lambda u: u["pol"]),
            "by_snr_pol": table(us, lambda u: f"{u['snr']:+.0f}/{u['pol']:+d}"),
            "by_lw": table(us, lambda u: u["lw"]),
            "by_L": table(us, lambda u: u["L_nominal"]),
            "by_L_short_line": table(us, lambda u: f"L{u['L_nominal']}|{'N<301' if u['short_line'] else 'N>=301'}"),
            "by_L_periods": table(us, lambda u: bin_of(u["L_periods_px"], lp_edges, lp_labels)),
            "by_survival": table(us, lambda u: bin_of(u["survival"], sv_edges, sv_labels)),
            "by_depth": table(us, lambda u: bin_of(u["apex_depth_P"], dep_edges, dep_labels)),
            "by_period_change": table(us, lambda u: bin_of(u["dp_rel"], dp_edges, dp_labels)),
            "by_period_changed_flag": table(us, lambda u: u["period_changed"]),
            "by_response": table(us, lambda u: u["response_id"]),
            "by_depth_snr": table(us, lambda u: f"{bin_of(u['apex_depth_P'], dep_edges, dep_labels)}|{u['snr']:+.0f}"),
            "by_L_snr": table(us, lambda u: f"L{u['L_nominal']:03d}|{u['snr']:+.0f}"),
            "by_y1px": table(us, lambda u: bin_of(u["truth_y1_px"], y_edges, y_labels)),
            "by_y1px_snr18": table(
                [u for u in us if u["snr"] == 18.0],
                lambda u: bin_of(u["truth_y1_px"], y_edges, y_labels),
            ),
            "by_inAIHub_L_y_snr18": table(
                [u for u in us if u["snr"] == 18.0],
                lambda u: f"L{'in' if u['L_periods_px'] >= 12.4 else 'short'}|y1{'>=37' if u['truth_y1_px'] >= 37 else '<37'}",
            ),
            "y1px_q": quantiles([u["truth_y1_px"] for u in us if u["fold"] == 0]),
            "by_hzone": table(us, lambda u: u["in_hzone"]),
            "by_fold": table(us, lambda u: u["fold"]),
            "survival_q": quantiles([u["survival"] for u in us if u["fold"] == 0]),
            "survival_q_by_snr": {
                str(snr): quantiles([u["survival"] for u in us if u["fold"] == 0 and u["snr"] == snr])
                for snr in SNRS
            },
            "h_P_q": quantiles([u["h_P"] for u in us if u["fold"] == 0]),
            "apex_depth_P_q": quantiles([u["apex_depth_P"] for u in us if u["fold"] == 0]),
            "L_periods_q": quantiles([u["L_periods_px"] for u in us if u["fold"] == 0]),
            "matched_h_P_q": quantiles([x for u in us for x in u["matched_h_P"]]),
            "matched_dc_P_q": quantiles([x for u in us for x in u["matched_dc_P"]]),
            "matched_w_over_L_q": quantiles([x for u in us for x in u["matched_w_over_L"]]),
            "matched_score_q": quantiles([x for u in us for x in u["matched_score"]]),
            "units_with_touching_box": sum(1 for u in us if u["n_touching"] > 0),
            "units_with_new_touching_box": sum(1 for u in us if u["n_touching_new"] > 0),
            "units_with_any_band_cover": sum(1 for u in us if u["cov_band50"] > 0),
        }
        # per line recall (pipe_band 50 %) and its concentration
        per_line = table(us, lambda u: u["line_id"], keys=("cov_band50", "cov_any50"))
        s["per_line"] = per_line
        vals = sorted(((v["cov_band50"] or 0.0), k) for k, v in per_line.items())
        s["lines_with_zero_band_recall"] = sum(1 for v, _ in vals if v == 0)
        # contribution of the top lines to the covered length
        cov_by_line = defaultdict(int)
        for u in us:
            cov_by_line[u["line_id"]] += u["cov_band50"]
        tot = sum(cov_by_line.values())
        top = sorted(cov_by_line.items(), key=lambda kv: -kv[1])
        s["covered_share_top_lines"] = [
            {"line_id": k, "share": round(v / tot, 4) if tot else None} for k, v in top[:6]
        ]
        # period-change comparison by POSITION (changed vs unchanged realizations
        # of the same position, any SNR / polarity; confounded by SNR because
        # changes happen mostly at 18 dB). The SNR-matched comparison is the
        # top-level ``period_change`` block (polarity twins).
        by_pos = defaultdict(lambda: {True: [], False: []})
        for u in us:
            by_pos[u["position_id"]][u["period_changed"]].append(u)
        pairs = [(v[True], v[False]) for v in by_pos.values() if v[True] and v[False]]
        if pairs:
            ch = [x for a, _ in pairs for x in a]
            un = [x for _, b in pairs for x in b]
            s["period_change_paired"] = {
                "n_positions": len(pairs),
                "changed_recall_band50": round(pooled(ch, "cov_band50"), 4),
                "unchanged_recall_band50": round(pooled(un, "cov_band50"), 4),
                "changed_recall_any50": round(pooled(ch, "cov_any50"), 4),
                "unchanged_recall_any50": round(pooled(un, "cov_any50"), 4),
                "changed_snr": sorted({x["snr"] for x in ch}),
            }
        out["scopes"][name] = s

    # Period change (|P_inj - P| > 0.05 P, change-report definition) per scope.
    # Every row of a scope uses that one scope: counts, simple comparison and
    # the polarity-twin comparison (same position, same SNR, one polarity
    # changed and the other not).
    def period_change_block(us) -> dict:
        f0 = [u for u in us if u["fold"] == FOLDS[0]]
        by_snr = {}
        for snr in SNRS:
            rows_snr = [u for u in f0 if u["snr"] == snr]
            by_snr[f"{snr:g}"] = {
                "changed_truths": sum(1 for u in rows_snr if u["period_changed"]),
                "truths": len(rows_snr),
            }
        ch_all = [u for u in us if u["period_changed"]]
        un_all = [u for u in us if not u["period_changed"]]
        cells = defaultdict(lambda: {True: [], False: []})
        for u in us:
            cells[(u["position_id"], u["snr"])][u["period_changed"]].append(u)
        twin_ch, twin_un, n_cells = [], [], 0
        for v in cells.values():
            if v[True] and v[False]:
                n_cells += 1
                twin_ch += v[True]
                twin_un += v[False]

        def r4(x):
            return None if x is None else round(x, 4)

        return {
            "changed_truths_fold0_by_snr": by_snr,
            "n_changed_realizations": len({u["realization_id"] for u in ch_all}),
            "n_changed_truths": len({u["injection_id"] for u in ch_all}),
            "n_changed_units": len(ch_all),
            "simple": {
                "changed_recall_band50": r4(pooled(ch_all, "cov_band50")),
                "unchanged_recall_band50": r4(pooled(un_all, "cov_band50")),
                "n_units_changed": len(ch_all),
                "n_units_unchanged": len(un_all),
            },
            "polarity_twins_same_position_same_snr": {
                "n_cells": n_cells,
                "n_units_changed": len(twin_ch),
                "n_units_unchanged": len(twin_un),
                "changed_recall_band50": r4(pooled(twin_ch, "cov_band50")),
                "unchanged_recall_band50": r4(pooled(twin_un, "cov_band50")),
            },
        }

    pc_scopes = {
        "p4": scopes["p4"],
        "ids_main400_main_cell": scopes["ids_main400_all_lw"],
        "ids_all_main_cell": [u for u in units if u["dataset"] == "gz_ids" and u["main_cell"]],
        "4tu_A0_main_cell": scopes["4tu_A0_main"],
        "4tu_A0_A1_main_cell": [u for u in units if u["dataset"] == "4tu" and u["main_cell"]],
    }
    out["period_change"] = {
        "definition": "changed = |P_inj - P_bg| > 0.05 P_bg (r3_change_report); recall = length-weighted pipe_band cover (50 % rule), pooled over truth x fold units",
        "scopes": {k: period_change_block(v) for k, v in pc_scopes.items()},
    }

    # un-injected pipe_band base rate and where those boxes sit (depth in P)
    base = {}
    for ds, line_ids in (
        ("gz_ids", {u["line_id"] for u in scopes["ids_main400_all_lw"]}),
        ("4tu", {u["line_id"] for u in units if u["dataset"] == "4tu"}),
    ):
        depth, per_line_counts, h_p = [], [], []
        cls_counts = defaultdict(int)
        for lid in line_ids:
            period = float(lines[lid]["period_bg_samples"])
            kz = float(lines[lid]["k_zero_bg"])
            n = 0
            for fold in FOLDS:
                for b in uninj.get((lid, fold), []):
                    cls_counts[b["cls"]] += 1
                    if b["cls"] != "pipe_band":
                        continue
                    n += 1
                    depth.append(((b["k0"] + b["k1"]) / 2 - kz) / period)
                    h_p.append((b["k1"] - b["k0"] + 1) / period)
            per_line_counts.append(n / len(FOLDS))
        base[ds] = {
            "n_lines": len(line_ids),
            "uninjected_boxes_by_class_5fold": dict(cls_counts),
            "band_boxes_per_line_per_fold_q": quantiles(per_line_counts),
            "lines_with_no_band_box": sum(1 for c in per_line_counts if c == 0),
            "band_box_center_depth_P_q": quantiles(depth),
            "band_box_h_P_q": quantiles(h_p),
        }
    out["uninjected_band_base"] = base

    # all pipe_band boxes in injected realizations of the P4 scope lines:
    # depth below zero (P) - do new band boxes appear only near the zero band?
    p4_real = {u["realization_id"] for u in scopes["p4"]}
    dep_all = []
    for (rid, fold), bs in inj_boxes.items():
        if rid not in p4_real:
            continue
        lid = rid.split(":")[0]
        period = float(lines[lid]["period_bg_samples"])
        kz = float(lines[lid]["k_zero_bg"])
        for b in bs:
            if b["cls"] == "pipe_band":
                dep_all.append(((b["k0"] + b["k1"]) / 2 - kz) / period)
    out["p4_injected_band_box_depth_P_q"] = quantiles(dep_all)

    # IDS lines with the largest period changes
    ids_dp = defaultdict(list)
    for r in real.values():
        if r["dataset"] == "gz_ids":
            ids_dp[r["line_id"]].append(float(r["period_dp_rel"]))
    out["ids_lines_max_abs_dp"] = sorted(
        ({"line_id": k, "max_abs_dp": round(max(abs(x) for x in v), 4), "n_gt5": sum(abs(x) > 0.05 for x in v)} for k, v in ids_dp.items()),
        key=lambda d: -d["max_abs_dp"],
    )[:8]

    with OUT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    cols = [k for k in units[0] if not k.startswith("matched_")]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for u in units:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in u.items() if k in cols})
    p4 = out["scopes"]["p4"]
    print(
        json.dumps(
            {
                "status": "ok",
                "n_units": len(units),
                "p4_recall_pre_check": p4["p4_style"]["cov_band50"],
                "out": str(OUT_JSON),
            }
        )
    )


if __name__ == "__main__":
    main()
