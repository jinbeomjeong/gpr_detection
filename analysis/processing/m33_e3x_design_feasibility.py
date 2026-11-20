"""M3-3 pre-closing (3) E3x design: feasibility counts from local plan tables.

Read only. Uses the M3-2 stage-C plan (hyperbola positions) and the R3 plan
(band positions, geometry per line); both are local files outside git (judged
material). Input sha256 values are written to the output. No detector output
is read. Writes
``analysis/processing/m33_e3x_design_feasibility.json``.

Run: conda run -n gpr-dev python -I -B analysis/processing/m33_e3x_design_feasibility.py
"""

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE_C_POS = REPO / "runs" / "eval" / "m32_recipe_v2" / "plan" / "positions.csv"
R3_POS = REPO / "runs" / "eval" / "m33" / "r3" / "plan" / "positions.csv"
R3_REPORT = REPO / "runs" / "eval" / "m33" / "r3" / "plan" / "plan_report.json"
IDS_LINES = REPO / "analysis" / "processing" / "m32_stageA_lines_ids.csv"
OUT = REPO / "analysis" / "processing" / "m33_e3x_design_feasibility.json"

Y1_EDGES = (37.0, 57.0, 79.0)
BIN_NAMES = ("lt37", "37_57", "57_79", "ge79")
LONG_P = 13.0
SHORT_P = 6.0
LW_MAX = 0.45
ZONE_B_P = 2.0
END_MARGIN_P = 1.0


def below_normal() -> None:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def q(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)

    def at(p: float) -> float:
        return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]

    return {"n": len(s), "min": s[0], "p05": at(0.05), "p50": at(0.5), "p95": at(0.95), "max": s[-1]}


def hyperbola_counts() -> dict:
    rows = read(STAGE_C_POS)
    per_model: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    lines: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["main_cell"] != "True":
            continue
        model = row["response_id"]
        key = (row["line_id"], row["apex_trace"])
        per_model[model][row["dataset"]].add(key)
        lines[model][row["dataset"]].add(row["line_id"])
    out = {}
    for model in sorted(per_model):
        out[model] = {
            ds: {"positions": len(per_model[model][ds]), "lines": len(lines[model][ds])}
            for ds in sorted(per_model[model])
        }
    return out


def band_geometry() -> dict:
    rows = read(R3_POS)
    windows = {r["line"]: float(r["time_window_ns"]) for r in read(IDS_LINES)}
    by_line: dict[str, dict] = {}
    offsets = []  # apex_px - y1_px per position (in L1 px)
    heights = []
    for row in rows:
        if row["main_cell"] != "True":
            continue
        period_px = float(row["period_px"])
        y1 = float(row["truth_y1_px"])
        y2 = float(row["truth_y2_px"])
        k0, k1 = float(row["k0"]), float(row["k1"])
        apex = float(row["apex_sample"])
        # px per sample from the truth box rows (L0 cell edges -> L1 px)
        px_per_sample = (y2 - y1) / (k1 - k0 + 1)
        apex_px = y1 + (apex - k0 + 0.5) * px_per_sample
        offsets.append((apex_px - y1) / period_px)
        heights.append((y2 - y1) / period_px)
        tpp = float(row["band_length_traces"]) / float(row["length_px"])
        line = by_line.setdefault(
            row["line_id"],
            {
                "dataset": row["dataset"],
                "stratum": row["ids_freq_stratum"],
                "n_traces": int(row["n_traces"]),
                "w_eff": int(row["w_eff"]),
                "period_px": period_px,
                "zero_px": float(row["zero_px"]),
                "y_bot_px": float(row["y_bot_px"]),
                "traces_per_px": tpp,
                "taper": int(row["taper_traces"]),
                "y1_seen": [],
            },
        )
        line["y1_seen"].append(y1)
    h_med = statistics.median(heights)
    off_med = statistics.median(offsets)
    bins = Counter()
    bins_long = Counter()
    bins_long_short = Counter()
    lines_out = {}
    for line_id, g in by_line.items():
        p = g["period_px"]
        y1_min = g["zero_px"] + (ZONE_B_P - off_med) * p
        y1_max = g["y_bot_px"] - END_MARGIN_P * p - h_med * p
        tpp = g["traces_per_px"]
        l_long = math.ceil(LONG_P * p * tpp)
        l_short = math.ceil(SHORT_P * p * tpp)
        long_ok = l_long <= LW_MAX * g["w_eff"] and l_long + 2 * g["taper"] <= g["n_traces"]
        short_ok = l_short + 2 * g["taper"] <= g["n_traces"] and l_short > g["taper"]
        reach = []
        edges = (-math.inf, *Y1_EDGES, math.inf)
        for name, lo, hi in zip(BIN_NAMES, edges[:-1], edges[1:]):
            if y1_max >= max(lo, y1_min) and min(hi, y1_max) >= y1_min and y1_max >= lo:
                reach.append(name)
        for name in reach:
            bins[(g["dataset"], g["stratum"], name)] += 1
            if long_ok:
                bins_long[(g["dataset"], g["stratum"], name)] += 1
                if short_ok:
                    bins_long_short[(g["dataset"], g["stratum"], name)] += 1
        lines_out[line_id] = {
            "dataset": g["dataset"],
            "stratum": g["stratum"],
            "time_window_ns": windows.get(line_id),
            "n_traces": g["n_traces"],
            "y1_min_px": y1_min,
            "y1_max_px": y1_max,
            "l_long_traces": l_long,
            "long_ok": long_ok,
            "short_ok": short_ok,
            "bins": reach,
            "all_four_bins": len(reach) == 4,
        }
    def table(counter: Counter) -> dict:
        return {f"{d}|{s}|{b}": n for (d, s, b), n in sorted(counter.items())}

    all4 = Counter()
    all4_long = Counter()
    for v in lines_out.values():
        k = f"{v['dataset']}|{v['stratum']}"
        if v["all_four_bins"]:
            all4[k] += 1
            if v["long_ok"]:
                all4_long[k] += 1
    three_inrange = Counter()
    for v in lines_out.values():
        need = {"lt37", "37_57", "57_79"}
        if need <= set(v["bins"]) and v["long_ok"]:
            three_inrange[f"{v['dataset']}|{v['stratum']}"] += 1
    return {
        "assumptions": {
            "truth_height_P_median": h_med,
            "apex_minus_y1_P_median": off_med,
            "y1_min": "zero_px + (2 P - (apex - y1)) ; zone (b)",
            "y1_max": "y_bot_px - 1 P - truth height ; C3-6 main cell",
            "long_P": LONG_P,
            "short_P": SHORT_P,
            "lw_max": LW_MAX,
        },
        "truth_height_P": q(heights),
        "apex_minus_y1_P": q(offsets),
        "lines_with_positions": len(lines_out),
        "bins_reachable_lines": table(bins),
        "bins_reachable_lines_long_ok": table(bins_long),
        "bins_reachable_lines_long_and_short_ok": table(bins_long_short),
        "lines_all_four_bins": dict(all4),
        "lines_all_four_bins_long_ok": dict(all4_long),
        "lines_three_low_bins_long_ok": dict(three_inrange),
        "ids_y1_max_px": q([v["y1_max_px"] for v in lines_out.values() if v["dataset"] == "gz_ids"]),
        "fourtu_y1_max_px": q([v["y1_max_px"] for v in lines_out.values() if v["dataset"] == "4tu"]),
        "lines": lines_out,
    }


TARGET_ROWS = (28.0, 47.0, 68.0, 90.0)
THREE_ROWS = (28.0, 47.0, 68.0)
SHIFT_LIMITS_P = (8.0, 12.0)


def shift_limit_eligibility(geometry: dict) -> dict:
    """Band shift-limit eligibility per R3 main-cell (line, model) draw.

    A draw is the R3 position's model on its line at native depth (truth_y1_px).
    Shift in P = |y* - y1_native| / period_px (image px). A target row is reachable
    on a line when y1_min <= y* <= y1_max (zone (b) and C3-6 main cell, same
    approximations as ``band_geometry``). Three-row eligibility also needs the long
    and short arms (selection step 3). This approximates the E3x selection rule;
    the R3 draws are a sample of library models per line, not the full library.
    """
    lines = geometry["lines"]
    rows = [r for r in read(R3_POS) if r["main_cell"] == "True"]
    out: dict = {
        "definition": {
            "draw": "R3 main-cell position (line, model) at native depth",
            "shift_P": "abs(target_px - truth_y1_px) / period_px",
            "row_reachable": "y1_min_px <= target_px <= y1_max_px (band_geometry lines)",
            "three_rows_ok": "rows 28,47,68 reachable and within limit, long_ok and short_ok",
            "four_rows_ok": "three_rows_ok and row 90 reachable and within limit",
            "native_range_nominal_P6": {
                f"{lim:g}P": {
                    "three_rows_native_y1_px": [68.0 - 6.0 * lim, 28.0 + 6.0 * lim],
                    "row90_native_y1_min_px": 90.0 - 6.0 * lim,
                }
                for lim in SHIFT_LIMITS_P
            },
        }
    }
    for lim in SHIFT_LIMITS_P:
        three = Counter()
        four = Counter()
        draws = Counter()
        models_three: dict[str, Counter] = defaultdict(Counter)
        models_four: dict[str, Counter] = defaultdict(Counter)
        lines_three: dict[str, set] = defaultdict(set)
        lines_four: dict[str, set] = defaultdict(set)
        shift_p_four: dict[str, list] = defaultdict(list)
        for row in rows:
            g = lines[row["line_id"]]
            key = f"{g['dataset']}|{g['stratum']}"
            draws[key] += 1
            y1 = float(row["truth_y1_px"])
            p = float(row["period_px"])

            def ok(target: float) -> bool:
                return g["y1_min_px"] <= target <= g["y1_max_px"] and abs(target - y1) / p <= lim

            if not (g["long_ok"] and g["short_ok"] and all(ok(t) for t in THREE_ROWS)):
                continue
            three[key] += 1
            models_three[key][row["response_id"]] += 1
            lines_three[key].add(row["line_id"])
            if ok(90.0):
                four[key] += 1
                models_four[key][row["response_id"]] += 1
                lines_four[key].add(row["line_id"])
                shift_p_four[key].extend(abs(t - y1) / p for t in TARGET_ROWS)
        out[f"limit_{lim:g}P"] = {
            "draws": dict(sorted(draws.items())),
            "draws_three_rows_ok": dict(sorted(three.items())),
            "draws_four_rows_ok": dict(sorted(four.items())),
            "frac_row90_given_three": {
                k: (four[k] / three[k] if three[k] else None) for k in sorted(three)
            },
            "lines_with_three_rows_draw": {k: len(v) for k, v in sorted(lines_three.items())},
            "lines_with_four_rows_draw": {k: len(v) for k, v in sorted(lines_four.items())},
            "models_three_rows_ok": {k: dict(v.most_common()) for k, v in sorted(models_three.items())},
            "models_four_rows_ok": {k: dict(v.most_common()) for k, v in sorted(models_four.items())},
            "shift_P_four_rows": {k: q(v) for k, v in sorted(shift_p_four.items())},
        }
    return out


def main() -> int:
    below_normal()
    report = R3_REPORT.exists()
    geometry = band_geometry()
    result = {
        "schema": "m33_e3x_design_feasibility/2",
        "inputs": [
            {"path": p.relative_to(REPO).as_posix(), "sha256": sha256_file(p)}
            for p in (STAGE_C_POS, R3_POS, IDS_LINES)
        ],
        "inputs_note": "runs/** plan tables are local (outside git), judged material; no detector output read",
        "r3_plan_report_present": report,
        "hyperbola_stage_c_main_cell": hyperbola_counts(),
        "band_r3_geometry": geometry,
        "band_shift_limit_eligibility": shift_limit_eligibility(geometry),
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "ok", "out": OUT.relative_to(REPO).as_posix()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
