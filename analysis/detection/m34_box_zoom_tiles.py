"""Zoomed AIHub tpipe box tiles for visual check of the M3-4 box convention (train_pool only).

Reads data/analysis/m34_box_convention/per_box.csv (from m34_aihub_tpipe_box_convention.py) and the crop images,
draws the label box (green), the automatic apex (red dot) and a 1-P scale bar, 5x nearest zoom, region = box
expanded by 1 box width / 1 box height. Output PNGs go to the git-ignored data/analysis/m34_box_convention/.
Example: conda run -n gpr-dev python -I analysis/detection/m34_box_zoom_tiles.py --n 12 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "aihub_yolo_v1"
OUT_DIR = ROOT / "data" / "analysis" / "m34_box_convention"
SPLIT_CSV = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="zoom")
    ap.add_argument("--sub", default="")
    ap.add_argument("--ticks", action="store_true", help="ticks at y0 -1.2/-0.6 P (cyan) and +0.6/+1.2 P (magenta)")
    args = ap.parse_args()
    import cv2
    import pandas as pd

    src_dir = OUT_DIR / args.sub if args.sub else OUT_DIR
    df = pd.read_csv(src_dir / "per_box.csv")
    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    split = split[split["subset"] == "trainpool"]
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(df), size=min(args.n, len(df)), replace=False))
    tiles = []
    z = 5
    for i in idx:
        r = df.iloc[i]
        sel = split[(split["crop_key"] == r["crop_key"]) & (split["gain"] == str(r["gain"]))]
        img = cv2.imread(str(DATASET / "images" / sel.iloc[0]["file_name"]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        bw, bh = r["x1"] - r["x0"], r["y1"] - r["y0"]
        X0, X1 = int(max(0, r["x0"] - 0.6 * bw)), int(min(w, r["x1"] + 0.6 * bw))
        Y0, Y1 = int(max(0, r["y0"] - 1.0 * bh)), int(min(h, r["y1"] + 0.6 * bh))
        sub = img[Y0:Y1, X0:X1]
        im = cv2.cvtColor(cv2.resize(sub, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im, (int((r["x0"] - X0) * z), int((r["y0"] - Y0) * z)),
                      (int((r["x1"] - X0) * z), int((r["y1"] - Y0) * z)), (0, 200, 0), 1)
        if args.ticks:
            Pq = r["P_px"] if not pd.isna(r["P_px"]) else 4.06
            xl, xr = int((r["x0"] - X0) * z), int((r["x1"] - X0) * z)
            for k, colr in ((-1.2, (255, 255, 0)), (-0.6, (255, 255, 0)), (0.6, (255, 0, 255)), (1.2, (255, 0, 255))):
                yy = int((r["y0"] + k * Pq - Y0) * z)
                cv2.line(im, (max(0, xl - 25), yy), (xl - 3, yy), colr, 1)
                cv2.line(im, (xr + 3, yy), (xr + 25, yy), colr, 1)
        elif r.get("apex_ok") and not pd.isna(r.get("apex_row")):
            cv2.circle(im, (int((r["apex_col"] - X0) * z), int((r["apex_row"] - Y0) * z)), 3, (0, 0, 255), -1)
        P = r["P_px"] if not pd.isna(r["P_px"]) else 4.06
        cv2.line(im, (3, 3), (3, 3 + int(P * z)), (255, 0, 255), 2)
        cv2.putText(im, f"#{i}", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(im)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0] + 6, 0, tw - t.shape[1] + 6, cv2.BORDER_CONSTANT,
                                value=(40, 40, 40)) for t in tiles]
    ncol = 4
    rows = []
    for k in range(0, len(tiles), ncol):
        row = tiles[k:k + ncol]
        row += [np.zeros_like(tiles[0])] * (ncol - len(row))
        rows.append(np.hstack(row))
    out = src_dir / f"{args.tag}_s{args.seed}.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(json.dumps({"status": "ok", "tiles": len(tiles), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
