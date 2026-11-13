"""QA-8 criterion 2 on real grids: detection boxes widen outward only, by < 1 cell per side.

For each of the 11 spec section 9 lines (same list as tests/test_detection_image_data.py):
  make_detection_image -> for every tile, random continuous L2 boxes (model px, not grid
  aligned, seed 20261113) -> letterbox transform of the real tile image -> L2 -> tile L1
  (model_box_to_image_box) -> line L1 (tile_box_to_line_box) -> LineBox -> line_boxes_to_grid
  (the M3-1 rounding path, once) -> GridBox.
The continuous L0 span is trace_offset + u * traces_per_px (and the same for samples).
Checks per side, for boxes whose continuous span lies inside the grid:
  inward:   i0 > t1 + tol  or  i1 + 1 < t2 - tol   (must be 0)
  widening: t1 - i0 >= 1   or  (i1 + 1) - t2 >= 1   (must be 0)
Boxes partly outside the grid are counted separately (clipping, criterion not applicable).
Also checks the exact grid-aligned round trip through the same LineBox path.
Read-only on data; writes analysis/processing/m31_qa8_outward_check.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\workspace\gpr_detection")
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.detect.base import PixelBox, letterbox  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.gprsd import DtHypothesis, load_gprsd_radargram  # noqa: E402
from gpr_detection.io.guangzhou import load_guangzhou_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.io.radargram import GridBox  # noqa: E402
from gpr_detection.pipeline import (  # noqa: E402
    GRID_ROUNDING_TOLERANCE,
    LineBox,
    grid_box_to_image_box,
    line_boxes_to_grid,
    model_box_to_image_box,
    tile_box_to_line_box,
)
from gpr_detection.preprocess.to_image import make_detection_image  # noqa: E402

SEED = 20261113
N_BOXES_PER_TILE = 60
MAX_TILES = 40  # long lines: every k-th tile so the run stays short
TOL = 1e-9
OUT = ROOT / "analysis" / "processing" / "m31_qa8_outward_check.json"

LINES = [
    ("4tu/03.9/Path5", DatasetName.FOURTU, lambda r: load_fourtu_radargram(r, "03.9", "Path5")),
    ("4tu/01.1/Path1", DatasetName.FOURTU, lambda r: load_fourtu_radargram(r, "01.1", "Path1")),
    ("4tu/09.1/Path2", DatasetName.FOURTU, lambda r: load_fourtu_radargram(r, "09.1", "Path2")),
    ("4tu/09.4/Path1", DatasetName.FOURTU, lambda r: load_fourtu_radargram(r, "09.4", "Path1")),
    ("gz/pipe/001_190321AA/LID10001", DatasetName.GUANGZHOU,
     lambda r: load_guangzhou_radargram(r, "pipe/001_190321AA/LID10001")),
    ("gz/pipe/1030_190117AF/LID10001", DatasetName.GUANGZHOU,
     lambda r: load_guangzhou_radargram(r, "pipe/1030_190117AF/LID10001")),
    ("gz/pipe/190321A_190321AH/LID10001", DatasetName.GUANGZHOU,
     lambda r: load_guangzhou_radargram(r, "pipe/190321A_190321AH/LID10001")),
    ("gz/pipe/MALA_07011/LA010001", DatasetName.GUANGZHOU,
     lambda r: load_guangzhou_radargram(r, "pipe/MALA_07011/LA010001")),
    ("gz/pipe/1030_181224AB/LID10001", DatasetName.GUANGZHOU,
     lambda r: load_guangzhou_radargram(r, "pipe/1030_181224AB/LID10001")),
    ("gprsd/CementRoad2_1", DatasetName.GPRSD,
     lambda r: load_gprsd_radargram(r, "CementRoad2_1", DtHypothesis.A)),
    ("gprsd/AsphaltRoad2_1", DatasetName.GPRSD,
     lambda r: load_gprsd_radargram(r, "AsphaltRoad2_1", DtHypothesis.A)),
]


def side_errors(start, stop, first, last):
    """(inward, too_wide, widen_lo, widen_hi) for one axis; cells are [j, j+1)."""
    lo = start - first
    hi = (last + 1) - stop
    inward = lo < -TOL - GRID_ROUNDING_TOLERANCE or hi < -TOL - GRID_ROUNDING_TOLERANCE
    too_wide = lo >= 1.0 or hi >= 1.0
    return inward, too_wide, lo, hi


def check_line(name, image, rng):
    gm = image.grid_map
    stats = {"line": name, "tiles": len(image.tiles), "tiles_checked": 0, "boxes": 0,
             "inside_grid": 0, "clipped_or_off": 0, "inward": 0, "too_wide": 0,
             "max_widen_trace": 0.0, "max_widen_sample": 0.0,
             "aligned_boxes": 0, "aligned_mismatch": 0,
             "traces_per_px": gm.traces_per_px, "samples_per_px": gm.samples_per_px,
             "overlap_px": image.overlap_px}
    step = max(1, math.ceil(len(image.tiles) / MAX_TILES))
    for tile in image.tiles[::step]:
        stats["tiles_checked"] += 1
        pixels = image.tile_image(tile.index)
        _, transform = letterbox(pixels, pixels.shape[0])
        for _ in range(N_BOXES_PER_TILE):
            # continuous L2 box inside the tile content (detector-like, not aligned)
            x1 = rng.uniform(0, tile.content_width_px)
            x2 = rng.uniform(x1, tile.content_width_px)
            y1 = rng.uniform(0, image.content_height_px)
            y2 = rng.uniform(y1, image.content_height_px)
            l2 = transform.apply(PixelBox(x1, y1, x2, y2))
            back = tile_box_to_line_box(model_box_to_image_box(l2, transform), tile)
            pairs = line_boxes_to_grid([LineBox(0, "pipe_hyperbola", 1.0, back, tile.index)], image)
            stats["boxes"] += 1
            t1 = gm.trace_offset + back.x1 * gm.traces_per_px
            t2 = gm.trace_offset + back.x2 * gm.traces_per_px
            s1 = gm.sample_offset + back.y1 * gm.samples_per_px
            s2 = gm.sample_offset + back.y2 * gm.samples_per_px
            inside = t1 >= 0 and s1 >= 0 and t2 <= gm.n_traces + 1e-9 and s2 <= gm.n_samples + 1e-9
            if not pairs or not inside:
                stats["clipped_or_off"] += 1
                continue
            stats["inside_grid"] += 1
            g = pairs[0][1]
            if t2 - t1 < 1e-9 or s2 - s1 < 1e-9:
                continue  # zero-width boxes cover one cell by contract
            a = side_errors(t1, t2, g.i0, g.i1)
            b = side_errors(s1, s2, g.k0, g.k1)
            stats["inward"] += a[0] or b[0]
            stats["too_wide"] += a[1] or b[1]
            stats["max_widen_trace"] = max(stats["max_widen_trace"], a[2], a[3])
            stats["max_widen_sample"] = max(stats["max_widen_sample"], b[2], b[3])
        # grid-aligned boxes fully inside this tile, through the same LineBox path
        lo_tr = math.ceil(tile.x0_px * gm.traces_per_px)
        hi_tr = math.floor((tile.x0_px + tile.content_width_px) * gm.traces_per_px) - 1
        if hi_tr < lo_tr:
            continue
        for _ in range(20):
            i0 = int(rng.integers(lo_tr, hi_tr + 1))
            i1 = int(rng.integers(i0, hi_tr + 1))
            k0 = int(rng.integers(0, gm.n_samples))
            k1 = int(rng.integers(k0, gm.n_samples))
            original = GridBox(i0, i1, k0, k1)
            lb = grid_box_to_image_box(original, gm)
            tb = PixelBox(lb.x1 - tile.x0_px, lb.y1, lb.x2 - tile.x0_px, lb.y2)
            back = tile_box_to_line_box(model_box_to_image_box(transform.apply(tb), transform), tile)
            pairs = line_boxes_to_grid([LineBox(0, "pipe_hyperbola", 1.0, back, tile.index)], image)
            stats["aligned_boxes"] += 1
            stats["aligned_mismatch"] += (not pairs) or pairs[0][1] != original
    stats["max_widen_trace"] = round(stats["max_widen_trace"], 6)
    stats["max_widen_sample"] = round(stats["max_widen_sample"], 6)
    return stats


def main():
    paths = load_dataset_paths()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, dataset, load in LINES:
        root = paths.root(dataset)
        image = make_detection_image(load(root))
        rows.append(check_line(name, image, rng))
        print(json.dumps(rows[-1]))
    total = {k: sum(r[k] for r in rows) for k in ("boxes", "inside_grid", "clipped_or_off",
                                                   "inward", "too_wide", "aligned_boxes", "aligned_mismatch")}
    total["max_widen_trace"] = max(r["max_widen_trace"] for r in rows)
    total["max_widen_sample"] = max(r["max_widen_sample"] for r in rows)
    OUT.write_text(json.dumps({"seed": SEED, "lines": rows, "total": total}, indent=1), encoding="utf-8")
    print(json.dumps({"total": total}))


if __name__ == "__main__":
    main()
