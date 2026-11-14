"""M3-2 stage A (S2): texture statistics of the recipes vs AIHub NM crops, shortlist (parameter selection).

Rules: docs/processing/m32_image_quality_plan.md v0.2.1 section 4-1 (recipes r0-r7 + report-only variants)
and section 4-2 (reference set, statistics, distance D_r, shortlist). Rules are not changed here; details the
plan leaves open are fixed below and disclosed in the report:
  * statistics on real lines use content rows only (rows < H_c), 256-px windows, <= 5 evenly spaced per line;
  * D_r is computed per real domain (4TU, Guangzhou IDS) and averaged with equal weight (like C-1);
    the pooled-window D_r is reported too;
  * G3 exponent n: the value that brings the median rho_bt of the 4TU S-bundle lines (inside the stage A list)
    under recipe r4 to the AIHub NM gain 1.0 median rho_bt (grid 0-3 step 0.25, linear interpolation, rounded
    to 0.05);
  * M2 reference CDF: pooled grey histogram of the AIHub NM gain 1.0 reference crops (written to
    m32_aihub_nm_cdf.json, used by r7).
Lists (deterministic, seed 20261114) are written as csv with sha256: AIHub NM reference (train_pool,
pipe_hyperbola / pipe_band, 300 crop keys each, gain 1.0 + the 0.25 / 0.5 copies of the same keys),
4TU stage A (bundles S, L, X of data/processing/d3/lines.csv minus K, plus site x dt proportional random fill to
150, byte-identical copies and K-bundle lines excluded), Guangzhou IDS (antenna x time-window proportional, 60).
No detector, no depth, no ground truth. Output: analysis/processing/m32_texture_recipes.json (+ list csvs).
Run: conda run -n gpr-dev python -I analysis/processing/m32_texture_recipes.py [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DEFAULT_DETECTION_IMAGE_CONFIG,
    make_detection_image,
)

SEED = 20261114
N_PER_CLASS = 300
N_4TU = 150
N_IDS = 60
WINDOWS_PER_LINE = 5
OUT_DIR = ROOT / "analysis" / "processing"
OUT = OUT_DIR / "m32_texture_recipes.json"
CDF_OUT = OUT_DIR / "m32_aihub_nm_cdf.json"
LIST_AIHUB = OUT_DIR / "m32_stageA_aihub_ref.csv"
LIST_4TU = OUT_DIR / "m32_stageA_lines_4tu.csv"
LIST_IDS = OUT_DIR / "m32_stageA_lines_ids.csv"
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
AIHUB_IMAGES = ROOT / "data" / "aihub_yolo_v1" / "images"
LINES_4TU = ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
DUPS = ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
D3_LINES = ROOT / "data" / "processing" / "d3" / "lines.csv"
SHAPE_STATS = ("R_h", "R_h_top15", "rho_bt", "flat_row_frac")
CONTRAST_STATS = ("std", "sat_frac")
NM_CLASSES = ("pipe_hyperbola", "pipe_band")
G3_GRID = tuple(np.round(np.arange(0.0, 3.01, 0.25), 2))

# Plan section 4-1. Values not listed keep the r0 defaults.
B0 = {"detection_bg_mode": "median"}
B1 = {"detection_bg_mode": "none"}
G0 = {"gain_mode": "agc"}
G2 = {"gain_mode": "agc_capped", "agc_floor_beta": 0.5}
P0 = {"pad_mode": "flat"}
P1 = {"pad_mode": "taper", "pad_taper_periods": 1.0}
RECIPES = {
    "r0": {},
    "r1": {**B1, **G2, **P1},
    "r2": {"detection_bg_mode": "partial", "detection_bg_alpha": 0.5, **G2, **P1},
    "r3": {"detection_bg_mode": "gated", "detection_bg_gate_periods": 2.0, **G0, **P1},
    "r4": {**B1, "gain_mode": "tpow", **P1},  # tpow_exponent filled after the G3 fit
    "r5": {**B0, **G2, **P1},
    "r6": {"detection_bg_mode": "partial", "detection_bg_alpha": 0.8, "gain_mode": "agc",
           "agc_window_periods": 32.0, **P0},
    "r7": {**B1, **G2, **P1, "gray_mapping": "hist_aihub"},  # cdf filled at run time
}
REPORT_ONLY = {
    "rep_B4_w1501": {"detection_bg_window": 1501},
    "rep_G4_none": {"gain_mode": "none"},
    "rep_P1_ntau2": {"pad_mode": "taper", "pad_taper_periods": 2.0},
    "rep_M1_p98": {"clip_percentile": 98.0},
    "rep_M1_p995": {"clip_percentile": 99.5},
}
M2_FAMILY = {"r7"}


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_list(path: Path, records: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proportional_pick(groups: dict[tuple, list], n_total: int, rng) -> list:
    """Largest-remainder proportional allocation over strata, random within each stratum."""
    keys = sorted(groups)
    size = sum(len(groups[k]) for k in keys)
    if size <= n_total:
        return [x for k in keys for x in groups[k]]
    quota = {k: n_total * len(groups[k]) / size for k in keys}
    alloc = {k: int(math.floor(quota[k])) for k in keys}
    rest = n_total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    out = []
    for k in keys:
        members = groups[k]
        idx = rng.choice(len(members), size=min(alloc[k], len(members)), replace=False)
        out.extend(members[i] for i in sorted(idx))
    return out


# ---------------------------------------------------------------- statistics
def stats(q: np.ndarray) -> dict[str, float | None]:
    q = np.asarray(q, dtype=np.float64)
    a = q - q.mean()
    total = float(np.sum(a**2))
    top = max(4, int(round(0.15 * q.shape[0])))
    at = a[:top] - a[:top].mean()
    tt = float(np.sum(at**2))
    s = q - 127.5
    half = q.shape[0] // 2
    rms_top = float(np.sqrt(np.mean(s[:half] ** 2)))
    rms_bot = float(np.sqrt(np.mean(s[half:] ** 2)))
    return {
        "R_h": q.shape[1] * float(np.sum(a.mean(axis=1) ** 2)) / total if total > 0 else None,
        "R_h_top15": q.shape[1] * float(np.sum(at.mean(axis=1) ** 2)) / tt if tt > 0 else None,
        "rho_bt": rms_bot / rms_top if rms_top > 0 else None,
        "flat_row_frac": float(np.mean(q.std(axis=1) < 2.0)),
        "std": float(q.std()),
        "sat_frac": float(np.mean((q <= 1) | (q >= 254))),
    }


def med_iqr(values) -> tuple[float, float, int]:
    """Median, spread and n. Spread = IQR; if the IQR is 0 (e.g. flat-row share mostly 0) the
    normal-equivalent (p95 - p5) * 1.349 / 3.29 is used, then 1e-3 (disclosed in the report)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    p5, q25, q50, q75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    spread = float(q75 - q25)
    if spread <= 0:
        spread = float((p95 - p5) * 1.349 / 3.29) or 1e-3
    return float(q50), spread, int(arr.size)


# ---------------------------------------------------------------- real lines
def load_line(ds: str, fid: str):
    paths = load_dataset_paths()
    if ds == "4tu":
        act, path = fid.split("/")
        return load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
    return load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), fid)


def make_config(recipe_id: str, overrides: dict) -> object:
    if not overrides:
        return DEFAULT_DETECTION_IMAGE_CONFIG
    return dataclasses.replace(DEFAULT_DETECTION_IMAGE_CONFIG, recipe_id=recipe_id, **overrides)


def line_windows(task: tuple) -> dict[str, object]:
    recipe_id, overrides, ds, fid = task
    cfg = make_config(recipe_id, overrides)
    img = make_detection_image(load_line(ds, fid), cfg)
    content = img.line_image[: img.content_height_px]
    width = content.shape[1]
    if width <= 256:
        starts = [0]
    else:
        starts = sorted(set(np.linspace(0, width - 256, WINDOWS_PER_LINE).round().astype(int).tolist()))
    return {"recipe": recipe_id, "dataset": ds, "line": fid, "flags": list(img.flags),
            "windows": [stats(content[:, s:s + 256]) for s in starts]}


def build_lists() -> tuple[list, list, list, dict]:
    rng = np.random.default_rng(SEED)
    # AIHub NM reference
    pool = [r for r in rows(SPLIT_CSV) if r["subset"] == "trainpool"]
    by_key: dict[str, dict[str, str]] = defaultdict(dict)
    for r in pool:
        by_key[r["crop_key"]][r["gain"]] = r["file_name"]
    ref = []
    for cls in NM_CLASSES:
        keys = sorted({r["crop_key"] for r in pool if r["primary_class"] == cls and r["gain"] == "1.0"})
        pick = rng.choice(len(keys), size=min(N_PER_CLASS, len(keys)), replace=False)
        for i in sorted(pick):
            k = keys[i]
            ref.append({"class": cls, "crop_key": k, "g1.0": by_key[k].get("1.0", ""),
                        "g0.5": by_key[k].get("0.5", ""), "g0.25": by_key[k].get("0.25", "")})
    # 4TU stage A
    dropped = {(r["activity_b"], r["path_b"]) for r in rows(DUPS)}
    d3 = {(r["activity_id"], r["path"]): r["bundles"] for r in rows(D3_LINES)}
    k_lines = {key for key, b in d3.items() if "K" in b}
    all4 = [r for r in rows(LINES_4TU) if r.get("npy_exists", "True") == "True"
            and (r["activity_id"], r["path"]) not in dropped and (r["activity_id"], r["path"]) not in k_lines]
    core = [r for r in all4 if any(x in d3.get((r["activity_id"], r["path"]), "") for x in ("S", "L", "X"))]
    core_keys = {(r["activity_id"], r["path"]) for r in core}
    rest = [r for r in all4 if (r["activity_id"], r["path"]) not in core_keys]
    groups: dict[tuple, list] = defaultdict(list)
    for r in rest:
        groups[(r["site"], r["dt_ns"])].append(r)
    fill = proportional_pick(groups, N_4TU - len(core), rng)
    l4 = [{"line": f"{r['activity_id']}/{r['path']}", "site": r["site"], "dt_ns": r["dt_ns"],
           "source": "SLX" if (r["activity_id"], r["path"]) in core_keys else "random"} for r in core + fill]
    # Guangzhou IDS
    gz_root = load_dataset_paths().root(DatasetName.GUANGZHOU)
    ids = [r for r in rows(gz_root / "01_메타데이터" / "radargrams_index.csv")
           if r["target"] == "pipe" and r["format"] != "MALA .rd3"]
    g2: dict[tuple, list] = defaultdict(list)
    for r in ids:
        g2[(r["antenna_MHz"], str(round(float(r["time_window_ns"]))))].append(r)
    lids = [{"line": r["file_id"], "antenna_MHz": r["antenna_MHz"],
             "time_window_ns": r["time_window_ns"]} for r in proportional_pick(g2, N_IDS, rng)]
    shas = {"aihub_ref": write_list(LIST_AIHUB, ref), "lines_4tu": write_list(LIST_4TU, l4),
            "lines_ids": write_list(LIST_IDS, lids)}
    return ref, l4, lids, shas


def aihub_reference(ref: list) -> tuple[dict, list[float]]:
    per_gain: dict[str, list] = {"1.0": [], "0.5": [], "0.25": []}
    hist = np.zeros(256, dtype=np.float64)
    for r in ref:
        for gain in per_gain:
            name = r[f"g{gain}"]
            if not name:
                continue
            q = np.asarray(Image.open(AIHUB_IMAGES / name).convert("L"))
            per_gain[gain].append(stats(q))
            if gain == "1.0":
                hist += np.bincount(q.ravel(), minlength=256)
    out = {}
    for gain, st in per_gain.items():
        out[gain] = {m: med_iqr([s[m] for s in st]) for m in SHAPE_STATS + CONTRAST_STATS}
        out[gain]["n_images"] = len(st)
        out[gain]["iqr_raw"] = {
            m: float(np.subtract(*np.percentile([s[m] for s in st if s[m] is not None], [75, 25])))
            for m in SHAPE_STATS + CONTRAST_STATS
        }
    cdf = np.cumsum(hist) / hist.sum()
    cdf[-1] = 1.0
    return out, [float(round(x, 10)) for x in cdf]


def distance(real_med: dict[str, float], ref: dict) -> tuple[float, dict[str, float]]:
    parts = {}
    for m in SHAPE_STATS:
        med, iqr, _ = ref["1.0"][m]
        parts[m] = abs(real_med[m] - med) / iqr
    for m in CONTRAST_STATS:
        parts[m] = min(abs(real_med[m] - ref[g][m][0]) / ref[g][m][1] for g in ("1.0", "0.5", "0.25"))
    return float(sum(parts.values())), parts


def summarise(results: list[dict], ref: dict) -> dict[str, object]:
    out = {}
    for dom in ("4tu", "gz_ids", "pooled"):
        wins = [w for r in results if dom == "pooled" or r["dataset"] == dom for w in r["windows"]]
        med = {m: med_iqr([w[m] for w in wins])[0] for m in SHAPE_STATS + CONTRAST_STATS}
        d, parts = distance(med, ref)
        out[dom] = {"median": med, "D": d, "parts": parts, "n_windows": len(wins)}
    out["D_r"] = 0.5 * (out["4tu"]["D"] + out["gz_ids"]["D"])
    out["n_lines"] = len(results)
    out["flag_counts"] = {f: sum(f in r["flags"] for r in results)
                          for f in ("detection_bg_gate_fallback", "period_fallback", "period_compressed")}
    return out


def run_recipe(pool, recipe_id: str, overrides: dict, lines: list[tuple[str, str]]) -> list[dict]:
    return pool.map(line_windows, [(recipe_id, overrides, ds, fid) for ds, fid in lines], chunksize=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ref_list, l4, lids, shas = build_lists()
    print(f"lists: aihub {len(ref_list)} 4tu {len(l4)} ids {len(lids)}", flush=True)
    ref, cdf = aihub_reference(ref_list)
    CDF_OUT.write_text(json.dumps({"source": "AIHub train_pool NM (pipe_hyperbola, pipe_band) gain 1.0",
                                   "list_sha256": shas["aihub_ref"], "n_images": ref["1.0"]["n_images"],
                                   "cdf": cdf}, indent=0), encoding="utf-8")
    lines = [("4tu", r["line"]) for r in l4] + [("gz_ids", r["line"]) for r in lids]
    s_lines = [("4tu", f"{r['activity_id']}/{r['path']}") for r in rows(D3_LINES)
               if "S" in r["bundles"] and "K" not in r["bundles"]]
    s_lines = [x for x in s_lines if x[1] in {r["line"] for r in l4}]
    target = ref["1.0"]["rho_bt"][0]
    result: dict[str, object] = {"seed": SEED, "list_sha256": shas, "aihub_reference": ref,
                                 "cdf_sha256": hashlib.sha256(CDF_OUT.read_bytes()).hexdigest()}
    with Pool(args.workers) as pool:
        grid = []
        for n in G3_GRID:
            res = run_recipe(pool, f"g3fit_{n}", {**RECIPES["r4"], "tpow_exponent": float(n)}, s_lines)
            grid.append(med_iqr([w["rho_bt"] for r in res for w in r["windows"]])[0])
            print(f"G3 n={n} rho_bt={grid[-1]:.3f}", flush=True)
        g = np.asarray(grid)
        if target <= g.min():
            n_fit = float(G3_GRID[int(np.argmin(g))])
        elif target >= g.max():
            n_fit = float(G3_GRID[int(np.argmax(g))])
        else:
            order = np.argsort(g)
            n_fit = float(np.interp(target, g[order], np.asarray(G3_GRID)[order]))
        n_fit = round(round(n_fit / 0.05) * 0.05, 2)
        result["g3_fit"] = {"target_rho_bt": target, "grid_n": list(G3_GRID), "grid_rho_bt": grid,
                            "n": n_fit, "lines": len(s_lines)}
        RECIPES["r4"]["tpow_exponent"] = n_fit
        RECIPES["r7"]["hist_reference_cdf"] = tuple(cdf)
        summary = {}
        for rid, ov in {**RECIPES, **REPORT_ONLY}.items():
            res = run_recipe(pool, rid, ov, lines)
            summary[rid] = summarise(res, ref)
            print(f"recipe {rid} done", flush=True)
    result["recipes"] = summary
    d0 = summary["r0"]["D_r"]
    cand = sorted((summary[r]["D_r"], r) for r in RECIPES if r != "r0" and summary[r]["D_r"] < 0.75 * d0)
    picked, m2 = [], 0
    for _, rid in cand:
        if rid in M2_FAMILY:
            if m2 >= 1:
                continue
            m2 += 1
        picked.append(rid)
        if len(picked) == 3:
            break
    pooled_rank = sorted((summary[r]["pooled"]["D"], r) for r in RECIPES)
    result["stage_a"] = {"D_r0": d0, "threshold": 0.75 * d0, "shortlist": ["r0", *picked],
                         "eligible_sorted": [r for _, r in cand],
                         "rank_by_D_r": [r for _, r in sorted((summary[r]["D_r"], r) for r in RECIPES)],
                         "rank_by_pooled_D": [r for _, r in pooled_rank],
                         "stage_c_needed": bool(picked)}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "shortlist": result["stage_a"]["shortlist"]}))


if __name__ == "__main__":
    main()
