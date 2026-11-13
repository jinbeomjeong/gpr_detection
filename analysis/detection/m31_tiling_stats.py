"""Statistics behind docs/experiments/m31_detection_input_tiling.md (M3-1).

Reads only:
  - AIHub YOLO labels of train_pool (data/aihub_yolo_v1, fold*_images.txt),
    crop unit = one gain copy per crop (1.0 > 0.5 > 0.25), to avoid counting
    gain copies as independent samples.
  - Line size tables of 4TU (docs/processing/fourtu_lines_v1.csv),
    Guangzhou and GPR-SD (radargrams_index.csv in GPR_Data, read-only).

Outputs a JSON summary (stdout last line + --out). No GPU, no images decoded.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AIHUB_DIR = REPO / "data" / "aihub_yolo_v1"
FOURTU_LINES = REPO / "docs" / "processing" / "fourtu_lines_v1.csv"
GPR_DATA = Path(r"C:\workspace\GPR_Data")
GZ_INDEX = GPR_DATA / "광저우대_중국_관로GPR_가공본_261016" / "01_메타데이터" / "radargrams_index.csv"
SD_INDEX = GPR_DATA / "GPR-SD_중국USTC_도로GPR_가공본_261016" / "01_메타데이터" / "radargrams_index.csv"
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
IMG_PX = 256
GAIN_RE = re.compile(r"_g(0\.25|0\.5|1\.0)_")
GAIN_PRIORITY = {"1.0": 0, "0.5": 1, "0.25": 2}
TILE_WIDTHS = (256, 512)
OVERLAP_FRACTIONS = (0.0, 0.25, 0.375, 0.5, 0.75)
PCTS = (5, 10, 25, 50, 75, 90, 95, 99)


def crop_representatives() -> list[Path]:
    """One label file per crop key (gain copy chosen by priority)."""
    best: dict[str, tuple[int, Path]] = {}
    for list_file in sorted(AIHUB_DIR.glob("fold*_images.txt")):
        for line in list_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            match = GAIN_RE.search(stem)
            if match is None:
                raise ValueError(f"no gain tag in {stem}")
            key = GAIN_RE.sub("_g*_", stem)
            prio = GAIN_PRIORITY[match.group(1)]
            label = AIHUB_DIR / "labels" / f"{stem}.txt"
            if key not in best or prio < best[key][0]:
                best[key] = (prio, label)
    return [path for _, path in best.values()]


def pct_table(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(values, p)), 2) for p in PCTS} | {
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
    }


def aihub_box_stats() -> dict:
    labels = crop_representatives()
    per_class: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for label in labels:
        if not label.is_file():
            continue
        for row in label.read_text(encoding="utf-8").splitlines():
            parts = row.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            w = float(parts[3]) * IMG_PX
            h = float(parts[4]) * IMG_PX
            yc = float(parts[2]) * IMG_PX
            xc = float(parts[1]) * IMG_PX
            per_class[cls].append((w, h, yc, xc))
    out = {"n_crops": len(labels), "classes": {}}
    for cls, rows in sorted(per_class.items()):
        arr = np.asarray(rows)
        w, h, yc, xc = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        x1, x2 = xc - w / 2, xc + w / 2
        touch = (x1 <= 2.0) | (x2 >= IMG_PX - 2.0)
        out["classes"][CLASS_NAMES[cls]] = {
            "n_boxes": int(len(arr)),
            "w_px": pct_table(w),
            "h_px": pct_table(h),
            "aspect_w_over_h": pct_table(w / np.maximum(h, 1e-6)),
            "y_bottom_px": pct_table(yc + h / 2),
            "frac_touch_lr_edge_2px": round(float(np.mean(touch)), 4),
            "x_center_px": pct_table(xc),
            "x_min_gap_to_lr_edge_px": pct_table(np.minimum(x1, IMG_PX - x2)),
            "frac_w_ge_200px": round(float(np.mean(w >= 200)), 4),
            "frac_w_ge_250px": round(float(np.mean(w >= 250)), 4),
            "frac_min_side_lt_8px": round(float(np.mean(np.minimum(w, h) < 8)), 4),
        }
    return out


X_RE = re.compile(r"^(?P<prefix>.+)_g(?:0\.25|0\.5|1\.0)_x\((?P<a>\d+)-(?P<b>\d+)\)$")


def aihub_crop_adjacency() -> dict:
    """Do crops of one session/channel abut or overlap along x?

    Adjacent or overlapping crops with labels on both sides would allow a
    stitched-line test bed for tile merging without new labels.
    """
    spans: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for list_file in sorted(AIHUB_DIR.glob("fold*_images.txt")):
        for line in list_file.read_text(encoding="utf-8").splitlines():
            stem = Path(line.strip()).stem
            match = X_RE.match(stem)
            if match:
                spans[match.group("prefix")].add((int(match.group("a")), int(match.group("b"))))
    n_pairs_overlap = n_pairs_abut = n_groups_multi = 0
    overlaps = []
    group_classes: dict[str, int] = defaultdict(int)
    group_crops: dict[str, int] = defaultdict(int)
    stitched_span = []
    for prefix, ranges in spans.items():
        ordered = sorted(ranges)
        if len(ordered) > 1:
            n_groups_multi += 1
            stitched_span.append(max(b for _, b in ordered) - ordered[0][0])
            present = set()
            for a, b in ordered:
                label = AIHUB_DIR / "labels" / f"{prefix}_g1.0_x({a}-{b}).txt"
                if not label.is_file():
                    candidates = list((AIHUB_DIR / "labels").glob(f"{glob_escape(prefix)}_g*_x({a}-{b}).txt"))
                    label = candidates[0] if candidates else label
                if label.is_file():
                    for row in label.read_text(encoding="utf-8").splitlines():
                        if row.strip():
                            present.add(CLASS_NAMES[int(row.split()[0])])
            for name in present:
                group_classes[name] += 1
                group_crops[name] += len(ordered)
        for (a0, b0), (a1, b1) in zip(ordered, ordered[1:]):
            if a1 < b0:
                n_pairs_overlap += 1
                overlaps.append(b0 - a1)
            elif a1 == b0 or a1 == b0 + 1:
                n_pairs_abut += 1
    return {
        "n_session_channel_groups": len(spans),
        "n_groups_with_multiple_crops": n_groups_multi,
        "n_consecutive_pairs_overlap": n_pairs_overlap,
        "n_consecutive_pairs_abut": n_pairs_abut,
        "overlap_px": pct_table(np.asarray(overlaps, dtype=float)) if overlaps else {},
        "multi_crop_groups_by_class": dict(group_classes),
        "multi_crop_group_crops_by_class": dict(group_crops),
        "stitched_span_px": pct_table(np.asarray(stitched_span, dtype=float)) if stitched_span else {},
    }


def glob_escape(text: str) -> str:
    """Escape glob metacharacters ([ ] * ?) in a literal file-name prefix."""
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def n_tiles(n_traces: int, width: int, overlap: float) -> int:
    if n_traces <= width:
        return 1
    stride = max(1, int(round(width * (1 - overlap))))
    return 1 + math.ceil((n_traces - width) / stride)


def line_stats(name: str, rows: list[dict], n_samples_key: str = "n_samples") -> dict:
    traces = np.asarray([int(float(r["n_traces"])) for r in rows if r.get("n_traces")])
    samples = np.asarray([int(float(r[n_samples_key])) for r in rows if r.get(n_samples_key)])
    traces = traces[traces > 0]
    tiles = {}
    for width in TILE_WIDTHS:
        for ov in OVERLAP_FRACTIONS:
            tiles[f"w{width}_ov{int(ov * 100)}"] = int(
                sum(n_tiles(int(t), width, ov) for t in traces)
            )
    # Option A: whole line letterboxed to 256: horizontal shrink factor
    shrink = traces / IMG_PX
    # Scale augmentation covers 0.8-1.2, so a whole-line letterbox stays in the
    # trained size range only while 256 / W >= 0.8, i.e. W <= 320 (1 trace/px).
    fit = {}
    for limit in (256, 320, 512, 1024):
        sel = traces <= limit
        fit[f"le{limit}"] = {
            "line_frac": round(float(np.mean(sel)), 4),
            "trace_frac": round(float(traces[sel].sum() / traces.sum()), 4),
        }
    return {
        "whole_line_fit": fit,
        "dataset": name,
        "n_lines": int(traces.size),
        "n_traces_total": int(traces.sum()),
        "n_traces": pct_table(traces.astype(float)),
        "n_samples_unique": sorted({int(s) for s in samples}),
        "letterbox_whole_line_x_shrink": pct_table(shrink.astype(float)),
        "tiles_total": tiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"aihub_train_pool_crop_unit": aihub_box_stats()}
    result["aihub_crop_adjacency"] = aihub_crop_adjacency()
    result["fourtu"] = line_stats("4TU", read_csv(FOURTU_LINES))
    gz_rows = [r for r in read_csv(GZ_INDEX) if r.get("target") == "pipe"]
    result["guangzhou_pipe"] = line_stats("Guangzhou pipe", gz_rows)
    result["gprsd"] = line_stats("GPR-SD", read_csv(SD_INDEX))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_crops": result["aihub_train_pool_crop_unit"]["n_crops"]}))


if __name__ == "__main__":
    main()
