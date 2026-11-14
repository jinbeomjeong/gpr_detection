"""M3-2 S2 checks after the S1 implementation (exploratory, not a verdict).

1. Plan section 2-5: re-run Part A of m32_baseline_texture.py (4TU dt 0.195 / 0.117 lines, default config = r0)
   with the S1 code and compare per line with the stored m32_baseline_texture.json (period_samples, H_c,
   compression, flags). The stored json is read only, not rewritten.
2. Plan section 1-5: the same comparison for m32_content_height_lines.csv (all 4TU + Guangzhou IDS lines).
3. AIHub pipe_hyperbola label box size in P units (P_A = 4.06 px, plan section 1-6) for the S4 in-distribution
   stratum: train_pool, gain 1.0 copies (one per crop key), YOLO labels class 0. Isotropic scale augmentation
   multiplies box size and period alike, so the distribution in P units is invariant under it.
Output: analysis/processing/m32_s1_recheck.json. Run: conda run -n gpr-dev python -I analysis/processing/m32_s1_recheck.py
"""

from __future__ import annotations

import csv
import importlib.util
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

HERE = Path(__file__).resolve().parent
OUT = HERE / "m32_s1_recheck.json"
P_A = 4.06
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
LABELS = ROOT / "data" / "aihub_yolo_v1" / "labels"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def part_a_compare(paths) -> dict[str, object]:
    base = load_module("m32_baseline_texture")
    stored = json.loads((HERE / "m32_baseline_texture.json").read_text(encoding="utf-8"))["part_a"]["lines"]
    fresh = base.part_a(paths)["lines"]
    old = {x["line"]: x for x in stored}
    diffs = []
    for x in fresh:
        o = old.get(x["line"])
        keys = ("period_samples", "content_height_px", "compression", "flags")
        if o is None or any(o[k] != x[k] for k in keys):
            diffs.append({"line": x["line"], "old": {k: o[k] for k in keys} if o else None,
                          "new": {k: x[k] for k in keys}})
    return {"n_lines": len(fresh), "n_stored": len(stored), "n_diff": len(diffs), "diffs": diffs[:20]}


def content_height_compare(paths) -> dict[str, object]:
    rows = list(csv.DictReader(open(HERE / "m32_content_height_lines.csv", encoding="utf-8")))
    diffs = 0
    examples = []
    for r in rows:
        if r["dataset"] == "4tu":
            act, path = r["line"].split("/")
            rg = load_fourtu_radargram(paths.root(DatasetName.FOURTU), act, path)
        else:
            rg = load_guangzhou_radargram(paths.root(DatasetName.GUANGZHOU), r["line"])
        img = make_detection_image(rg)
        new = (round(img.period_samples, 4), img.content_height_px, img.line_image.shape[1], len(img.tiles),
               round(img.compression, 4))
        old = (float(r["period_samples"]), int(r["H_c"]), int(r["W1"]), int(r["n_tiles"]), float(r["compression"]))
        if new != old:
            diffs += 1
            if len(examples) < 10:
                examples.append({"line": r["line"], "old": old, "new": new})
    return {"n_lines": len(rows), "n_diff": diffs, "examples": examples}


def aihub_box_p() -> dict[str, object]:
    pool = [r for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8-sig"))
            if r["subset"] == "trainpool" and r["gain"] == "1.0"]
    w, h = [], []
    for r in pool:
        path = LABELS / (Path(r["file_name"]).stem + ".txt")
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) == 5 and parts[0] == "0":
                w.append(float(parts[3]) * 256 / P_A)
                h.append(float(parts[4]) * 256 / P_A)
    def pct(v):
        return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in (5, 10, 50, 90, 95)} | {"n": len(v)}
    return {"P_A_px": P_A, "width_P": pct(w), "height_P": pct(h),
            "note": "train_pool gain 1.0 copies, class 0 pipe_hyperbola, YOLO normalised box x 256 / P_A"}


def main() -> None:
    paths = load_dataset_paths()
    result = {"part_a": part_a_compare(paths), "content_height": content_height_compare(paths),
              "aihub_hyperbola_box_P": aihub_box_p()}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "part_a_diff": result["part_a"]["n_diff"],
                      "content_height_diff": result["content_height"]["n_diff"]}))


if __name__ == "__main__":
    main()
