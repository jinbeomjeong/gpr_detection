"""M3-2 plan pilot (exploratory, not a verdict): baseline numbers for the image-quality plan.

Part A  4TU dt 0.195 ns lines (83) and dt 0.117 ns (2): measured P_line, content height H_c and
        compression with the implemented default DetectionImage (P* 6). Compression starts when
        round(n_s / (P_line / P*)) > 256, i.e. P_line < n_s * P* / 256.5 samples.
Part B  Gray-level texture statistics of AIHub train_pool crops (all three gains, test excluded)
        vs real C' tiles / line content (4TU D3 bundle S, the 11 spec section 9 lines).
        Per image: mean, std, saturated fraction (q <= 1 or q >= 254), near-128 fraction
        (|q - 128| <= 2), flat-row fraction (row std < 2) and the median grey of flat rows,
        lower/upper half RMS ratio of (q - 127.5) (depth equalisation index), whole-image period
        (same estimator as measure_period_samples on q - 127.5, min period 2 px).
No detector, no depth, no 4TU ground truth. Seed 0 for the AIHub sample.

Output: analysis/processing/m32_baseline_texture.json. stdout last line: JSON summary.
Run: conda run -n gpr-dev python -I analysis/processing/m32_baseline_texture.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.gprsd import DtHypothesis, load_gprsd_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DEFAULT_DETECTION_IMAGE_CONFIG,
    make_detection_image,
    measure_period_samples,
)

SEED = 0
N_PER_GAIN = 100
TILES_PER_LINE = 5
LINES_CSV = ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
D3_LINES = ROOT / "data" / "processing" / "d3" / "lines.csv"
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
AIHUB_IMAGES = ROOT / "data" / "aihub_yolo_v1" / "images"
OUT = ROOT / "analysis" / "processing" / "m32_baseline_texture.json"
QA_LINES = (
    ("fourtu", "03.9/Path5"), ("fourtu", "01.1/Path1"), ("fourtu", "09.1/Path2"),
    ("fourtu", "09.4/Path1"),
    ("guangzhou", "pipe/001_190321AA/LID10001"), ("guangzhou", "pipe/1030_190117AF/LID10001"),
    ("guangzhou", "pipe/190321A_190321AH/LID10001"), ("guangzhou", "pipe/MALA_07011/LA010001"),
    ("guangzhou", "pipe/1030_181224AB/LID10001"),
    ("gprsd", "CementRoad2_1"), ("gprsd", "AsphaltRoad2_1"),
)


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(values, qs=(5, 25, 50, 75, 95)) -> dict[str, float] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    out = {f"p{q}": round(float(np.percentile(arr, q)), 4) for q in qs}
    out["n"] = int(arr.size)
    return out


def image_stats(q: np.ndarray) -> dict[str, float | None]:
    q = np.asarray(q, dtype=np.float64)
    a = q - 127.5
    row_std = q.std(axis=1)
    flat = row_std < 2.0
    half = q.shape[0] // 2
    rms_top = float(np.sqrt(np.mean(a[:half] ** 2)))
    rms_bot = float(np.sqrt(np.mean(a[half:] ** 2)))
    period = measure_period_samples(a, min_period=2.0) if q.shape[0] >= 16 else None
    return {
        "mean": float(q.mean()),
        "std": float(q.std()),
        "sat_frac": float(np.mean((q <= 1) | (q >= 254))),
        "near128_frac": float(np.mean(np.abs(q - 128.0) <= 2.0)),
        "flat_row_frac": float(flat.mean()),
        "flat_row_grey": float(np.median(q[flat].mean(axis=1))) if flat.any() else None,
        "rms_bottom_over_top": rms_bot / rms_top if rms_top > 0 else None,
        "period_px": period,
    }


def summarise(stats: list[dict[str, float | None]]) -> dict[str, object]:
    keys = stats[0].keys() if stats else ()
    return {k: pct([s[k] for s in stats]) for k in keys} | {"n_images": len(stats)}


def part_a(paths) -> dict[str, object]:
    root = paths.root(DatasetName.FOURTU)
    cfg = DEFAULT_DETECTION_IMAGE_CONFIG
    out = []
    for r in rows(LINES_CSV):
        dt = float(r["dt_ns"])
        if dt not in (0.195, 0.117) or r.get("npy_exists", "True") != "True":
            continue
        rg = load_fourtu_radargram(root, r["activity_id"], r["path"])
        img = make_detection_image(rg, cfg)
        n_s = rg.data.shape[0]
        p_c = n_s * cfg.p_target_px / (cfg.tile_px + 0.5)
        out.append({
            "line": f"{r['activity_id']}/{r['path']}", "dt_ns": dt, "n_traces": int(r["n_traces"]),
            "period_samples": img.period_samples, "period_ns": img.period_samples * dt,
            "period_source": img.period_source, "content_height_px": img.content_height_px,
            "compression": img.compression, "compress_threshold_samples": p_c,
            "margin_rel": img.period_samples / p_c - 1.0, "flags": list(img.flags),
        })
        print(f"A {out[-1]['line']} P={img.period_samples:.2f} H={img.content_height_px}", flush=True)
    sub = [o for o in out if o["dt_ns"] == 0.195]
    return {
        "lines": out,
        "dt0195": {
            "n": len(sub),
            "period_samples": pct([o["period_samples"] for o in sub], (0, 5, 25, 50, 75, 95, 100)),
            "period_ns": pct([o["period_ns"] for o in sub], (0, 5, 50, 95, 100)),
            "content_height_px": pct([o["content_height_px"] for o in sub], (0, 50, 95, 100)),
            "n_compressed": sum(o["compression"] > 1.0 for o in sub),
            "n_height_ge_240": sum(o["content_height_px"] >= 240 for o in sub),
            "n_margin_lt_5pct": sum(o["margin_rel"] < 0.05 for o in sub),
            "n_period_fallback": sum("period_fallback" in o["flags"] for o in sub),
            "compress_threshold_samples": sub[0]["compress_threshold_samples"] if sub else None,
        },
        "note": "duplicates (docs/datasets/fourtu_duplicates.md) not removed",
    }


def part_b_aihub() -> dict[str, object]:
    pool = [r for r in rows(SPLIT_CSV) if r["subset"] == "trainpool"]
    rng = np.random.default_rng(SEED)
    out: dict[str, object] = {}
    for gain in ("0.25", "0.5", "1.0"):
        cand = [r for r in pool if r["gain"] == gain]
        pick = rng.choice(len(cand), size=min(N_PER_GAIN, len(cand)), replace=False)
        stats = []
        for i in sorted(pick):
            q = np.asarray(Image.open(AIHUB_IMAGES / cand[i]["file_name"]).convert("L"))
            stats.append(image_stats(q))
        out[f"gain_{gain}"] = summarise(stats)
    return out


def real_lines(paths) -> list[tuple[str, object]]:
    lines = []
    for r in rows(D3_LINES):
        if "S" in r["bundles"]:
            lines.append((f"4tu/{r['activity_id']}/{r['path']}", ("fourtu", f"{r['activity_id']}/{r['path']}")))
    seen = {name for name, _ in lines}
    for ds, fid in QA_LINES:
        name = f"{ds}/{fid}"
        if name.replace("fourtu", "4tu") not in seen:
            lines.append((name, (ds, fid)))
    return lines


def load(paths, ds: str, fid: str):
    if ds == "fourtu":
        act, path = fid.split("/")
        return load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
    if ds == "guangzhou":
        return load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), fid)
    return load_gprsd_radargram(paths.root(DatasetName.GPRSD), fid, DtHypothesis.A)


def part_b_real(paths) -> dict[str, object]:
    groups: dict[str, dict[str, list]] = {}
    per_line = []
    for name, (ds, fid) in real_lines(paths):
        img = make_detection_image(load(paths, ds, fid))
        key = "4tu" if ds == "fourtu" else ("gz_mala" if "MALA" in fid else ds)
        g = groups.setdefault(key, {"tile": [], "content": []})
        n = len(img.tiles)
        ks = sorted(set(np.linspace(0, n - 1, min(TILES_PER_LINE, n)).round().astype(int).tolist()))
        for k in ks:
            g["tile"].append(image_stats(img.tile_image(k)))
        content = img.line_image[: img.content_height_px]
        cs = image_stats(content[:, : min(content.shape[1], 4096)])
        g["content"].append(cs)
        per_line.append({"line": name, "H_c": img.content_height_px, "W1": img.line_image.shape[1],
                         "pad_frac_rows": 1.0 - img.content_height_px / 256.0,
                         "content_period_px": cs["period_px"], "period_l1_px": img.period_l1_px,
                         "rms_bottom_over_top": cs["rms_bottom_over_top"], "sat_frac": cs["sat_frac"]})
        print(f"B {name} H={img.content_height_px} Pwhole={cs['period_px']}", flush=True)
    return {"groups": {k: {kk: summarise(vv) for kk, vv in v.items()} for k, v in groups.items()},
            "lines": per_line}


def main() -> None:
    paths = load_dataset_paths()
    result = {"seed": SEED, "config": {"p_target_px": DEFAULT_DETECTION_IMAGE_CONFIG.p_target_px,
                                       "pad_gray": DEFAULT_DETECTION_IMAGE_CONFIG.pad_gray,
                                       "detection_bg_window": DEFAULT_DETECTION_IMAGE_CONFIG.detection_bg_window}}
    result["part_a"] = part_a(paths)
    result["part_b_aihub"] = part_b_aihub()
    result["part_b_real"] = part_b_real(paths)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    a = result["part_a"]["dt0195"]
    print(json.dumps({"out": str(OUT), "dt0195_n": a["n"], "n_compressed": a["n_compressed"],
                      "n_margin_lt_5pct": a["n_margin_lt_5pct"]}))


if __name__ == "__main__":
    main()
