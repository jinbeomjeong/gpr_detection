"""M3-2 plan pilot (exploratory, not a verdict): horizontally coherent energy, AIHub vs real C' content.

Question for Q5 (detection background window): do AIHub training crops keep horizontally coherent
energy (layers, bands, surface band) that a 301-trace moving median removes from real lines?

Index per image region: R_h = sum_k W * mean_x(a)[k]^2 / sum a^2 with a = q - mean(q) (q uint8),
i.e. the fraction of energy in the row-mean (perfectly horizontal) component. Reported for the whole
image and for the top 15 % of rows (surface / direct-wave zone). AIHub: train_pool, seed 0, 100 crops per
primary class (gain 1.0 copies only). Real: 4TU D3 bundle S lines, content rows only, in 256-px-wide
windows (up to 5 evenly spaced windows per line), with detection_bg_window 301 (default) and 100001
(longer than every line = whole-line median).
No detector, no depth, no ground truth. Output: analysis/processing/m32_horizontal_coherence.json.
Run: conda run -n gpr-dev python -I analysis/processing/m32_horizontal_coherence.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DEFAULT_DETECTION_IMAGE_CONFIG,
    make_detection_image,
)

SEED = 0
N_PER_CLASS = 100
WINDOWS = (301, 100001)
D3_LINES = ROOT / "data" / "processing" / "d3" / "lines.csv"
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
AIHUB_IMAGES = ROOT / "data" / "aihub_yolo_v1" / "images"
OUT = ROOT / "analysis" / "processing" / "m32_horizontal_coherence.json"


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def r_h(q: np.ndarray) -> float | None:
    a = np.asarray(q, dtype=np.float64)
    a = a - a.mean()
    total = float(np.sum(a**2))
    if total <= 0:
        return None
    row_mean = a.mean(axis=1)
    return float(a.shape[1] * np.sum(row_mean**2) / total)


def pct(values) -> dict[str, float] | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(arr, q)), 4) for q in (10, 25, 50, 75, 90)} | {"n": int(arr.size)}


def region_pair(q: np.ndarray) -> tuple[float | None, float | None]:
    top = max(4, int(round(0.15 * q.shape[0])))
    return r_h(q), r_h(q[:top])


def aihub() -> dict[str, object]:
    pool = [r for r in rows(SPLIT_CSV) if r["subset"] == "trainpool" and r["gain"] == "1.0"]
    rng = np.random.default_rng(SEED)
    out: dict[str, object] = {}
    for cls in ("pipe_hyperbola", "pipe_band", "cavity", "manhole"):
        cand = [r for r in pool if r["primary_class"] == cls]
        pick = rng.choice(len(cand), size=min(N_PER_CLASS, len(cand)), replace=False)
        whole, top = [], []
        for i in sorted(pick):
            q = np.asarray(Image.open(AIHUB_IMAGES / cand[i]["file_name"]).convert("L"))
            w, t = region_pair(q)
            whole.append(w)
            top.append(t)
        out[cls] = {"whole": pct(whole), "top15": pct(top)}
    return out


def real(paths) -> dict[str, object]:
    root = paths.root(DatasetName.FOURTU)
    lines = [r for r in rows(D3_LINES) if "S" in r["bundles"]]
    out: dict[str, object] = {}
    for window in WINDOWS:
        cfg = dataclasses.replace(DEFAULT_DETECTION_IMAGE_CONFIG, detection_bg_window=window)
        whole, top = [], []
        for r in lines:
            img = make_detection_image(load_fourtu_radargram(root, r["activity_id"], r["path"]), cfg)
            content = img.line_image[: img.content_height_px]
            width = content.shape[1]
            starts = [0] if width <= 256 else sorted(set(np.linspace(0, width - 256, 5).round().astype(int).tolist()))
            for s in starts:
                w, t = region_pair(content[:, s : s + 256])
                whole.append(w)
                top.append(t)
        out[f"bg_{window}"] = {"whole": pct(whole), "top15": pct(top), "n_lines": len(lines)}
        print(f"window {window} done", flush=True)
    return out


def main() -> None:
    result = {"seed": SEED, "aihub_gain_1.0": aihub(), "fourtu_S_content": real(load_dataset_paths())}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(OUT)}))


if __name__ == "__main__":
    main()
