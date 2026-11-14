"""M3-2 plan v0.2 input (exploratory, not a verdict): real C' content height H_c and bottom pad height.

Requested by cv-detection-expert (docs/experiments/m32_cv_review.md section 4-1 S3 item (2)): the bottom
pad height distribution of real L1 tiles at the default config (P* 6) for the AIHub pad perturbation.
Lines: all 4TU radargrams (byte-identical copies, docs/datasets/fourtu_duplicate_radargrams.csv file_b,
dropped) and all Guangzhou IDS pipe lines (MALA excluded: amplitude_processed, R1 open).
Per line: P_line, period_source, H_c, pad = 256 - H_c, W1, tile count, compression.
Distributions are given line-weighted and tile-weighted (each tile carries its line's pad).
Geometry only: no detector, no depth, no ground truth.
Output: analysis/processing/m32_content_height.json (+ per-line csv m32_content_height_lines.csv).
Run: conda run -n gpr-dev python -I analysis/processing/m32_content_height.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

LINES_4TU = ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
DUPS = ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
OUT = ROOT / "analysis" / "processing" / "m32_content_height.json"
OUT_CSV = ROOT / "analysis" / "processing" / "m32_content_height_lines.csv"
QS = (0, 5, 10, 25, 50, 75, 90, 95, 100)


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dist(values: list[float], weights: list[int] | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if weights is not None:
        arr = np.repeat(arr, np.asarray(weights, dtype=np.int64))
    out = {f"p{q}": round(float(np.percentile(arr, q)), 2) for q in QS}
    out["n"] = int(arr.size)
    out["frac_pad_gt_128"] = round(float(np.mean(arr > 128)), 4) if arr.size else None
    return out


def main() -> None:
    paths = load_dataset_paths()
    dropped = {(r["activity_b"], r["path_b"]) for r in rows(DUPS)}
    jobs = []
    for r in rows(LINES_4TU):
        if r.get("npy_exists", "True") != "True" or (r["activity_id"], r["path"]) in dropped:
            continue
        jobs.append(("4tu", f"{r['activity_id']}/{r['path']}", r["dt_ns"]))
    gz_root = paths.root(DatasetName.GUANGZHOU)
    for r in rows(gz_root / "01_메타데이터" / "radargrams_index.csv"):
        if r["target"] == "pipe" and r["format"] != "MALA .rd3":
            jobs.append(("gz_ids", r["file_id"], r["dt_ns"]))
    per_line = []
    for i, (ds, fid, dt) in enumerate(jobs):
        if ds == "4tu":
            act, path = fid.split("/")
            rg = load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
        else:
            rg = load_guangzhou_radargram(gz_root, fid)
        img = make_detection_image(rg)
        per_line.append({
            "dataset": ds, "line": fid, "dt_ns": dt, "period_samples": round(img.period_samples, 4),
            "period_source": img.period_source, "H_c": img.content_height_px,
            "pad_px": 256 - img.content_height_px, "W1": img.line_image.shape[1],
            "n_tiles": len(img.tiles), "compression": round(img.compression, 4),
        })
        if i % 50 == 0:
            print(f"{i}/{len(jobs)}", flush=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_line[0]))
        writer.writeheader()
        writer.writerows(per_line)
    result = {"note": "4TU byte-identical copies (file_b) dropped; Guangzhou MALA excluded"}
    for ds in ("4tu", "gz_ids"):
        sub = [p for p in per_line if p["dataset"] == ds]
        result[ds] = {
            "n_lines": len(sub), "n_tiles": int(sum(p["n_tiles"] for p in sub)),
            "H_c_line": dist([p["H_c"] for p in sub]),
            "pad_line": dist([p["pad_px"] for p in sub]),
            "pad_tile": dist([p["pad_px"] for p in sub], [p["n_tiles"] for p in sub]),
            "n_compressed": sum(p["compression"] > 1.0 for p in sub),
            "n_period_fallback": sum(p["period_source"] != "measured" for p in sub),
        }
    both = [p for p in per_line]
    result["4tu_plus_gz_ids"] = {
        "pad_line": dist([p["pad_px"] for p in both]),
        "pad_tile": dist([p["pad_px"] for p in both], [p["n_tiles"] for p in both]),
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n_lines": len(per_line)}))


if __name__ == "__main__":
    main()
