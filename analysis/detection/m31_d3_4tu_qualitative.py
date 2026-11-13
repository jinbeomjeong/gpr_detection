"""D3: qualitative 4TU inference through the C' image, tiles and merge (spec v0.4 §13).

No verdict, no depth: the script never calls a depth function and never opens
4TU ground-truth files. Known apex positions come from the blind review labels
(``docs/processing/m29c_apex_review.csv``, type T; only id, line, trace and
sample columns are read).

Steps
1. Select lines deterministically (§13-3): bundles S, K, L, X from
   ``docs/processing/fourtu_lines_v1.csv`` with the duplicate copies of
   ``docs/datasets/fourtu_duplicate_radargrams.csv`` removed by the keep_one
   rule (strict line-set superset, else the lower activity number; within an
   activity the lower path number). ``lines.csv`` is written first and hashed.
2. Per line: ``make_detection_image`` (default config, P* 6, O 192) once, then
   for each N2-a' fold model (``last.pt``, independent, no ensemble):
   ``detect_tiles`` -> ``merge_line_boxes`` -> ``line_boxes_to_grid``.
3. Measurements of §13-4 (boxes, V1/V2 pairs, truncation, band R3 geometry,
   flags, known apexes, cross-fold agreement of fold 0) and a fold-0 gallery.

Outputs go to ``--out-dir`` (default ``data/processing/d3``, git-ignored); an
existing non-empty folder is refused. All text files are UTF-8 with LF.

Example (GPU, all lines and folds)::

    python -I -B analysis/detection/m31_d3_4tu_qualitative.py --device cuda:0
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gpr_detection.depth.band import candidate_pairs
from gpr_detection.depth.config import DEFAULT_BAND_CONFIG
from gpr_detection.detect.yolo import load_yolo_detector
from gpr_detection.io.classes import DetectionClass
from gpr_detection.io.fourtu import load_fourtu_radargram
from gpr_detection.io.paths import DatasetName, load_dataset_paths
from gpr_detection.io.radargram import GridBox
from gpr_detection.pipeline import (
    DEFAULT_MERGE_CONFIG,
    LineBox,
    _fresnel_radius_px,  # same V2 radius as the merge (no public alias)
    _is_r2_duplicate,  # same R2 test as the merge (no public alias)
    detect_tiles,
    line_boxes_to_grid,
    merge_line_boxes,
)
from gpr_detection.preprocess.to_image import (
    DEFAULT_DETECTION_IMAGE_CONFIG,
    DetectionImage,
    make_detection_image,
)
from gpr_detection.run_info import (
    git_state,
    local_timestamp,
    package_versions,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINES_CSV = PROJECT_ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
DUPLICATES_CSV = PROJECT_ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
APEX_REVIEW_CSV = PROJECT_ROOT / "docs" / "processing" / "m29c_apex_review.csv"
APEX_REVIEW_COLUMNS = ("blind_id", "activity_id", "path", "trace", "sample", "type")
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processing" / "d3"
WEIGHTS_PATTERN = "runs/detect/yolo11m_aihubv1n2a2_20261114_fold{fold}/weights/last.pt"
FOLDS = (0, 1, 2, 3, 4)
MAIN_FOLD = 0
"""Median-mAP50 fold of the A1 judgement, fixed before looking at 4TU (§13-2)."""

CONFIDENCE = 0.25
AGREE_IOU = 0.5
APEX_TRACE_MARGIN = 3
APEX_PERIOD_MARGIN = 0.5
V1_NEAR_RANGE_P = (0.5, 1.5)
S_LINES_PER_SITE = 2
L_LONGEST = 3
X_DT_NS = 0.117
X_EPS_VALUES = (19.46, 17.36)
GALLERY_CHUNK_PX = 1024
GALLERY_SCALE = 2
WITHIN_ACTIVITY = "within_activity"
ACROSS_ACTIVITIES = "across_activities"

CLASS_COLORS_BGR = {
    DetectionClass.PIPE_HYPERBOLA: (0, 200, 255),
    DetectionClass.PIPE_BAND: (255, 160, 0),
    DetectionClass.CAVITY: (0, 220, 0),
    DetectionClass.MANHOLE: (220, 0, 220),
}
SEAM_COLOR_BGR = (90, 90, 90)
VETO_COLOR_BGR = (0, 0, 255)
APEX_COLOR_BGR = (255, 255, 255)
TRUNCATED_THICKNESS = 3

LOGGER = logging.getLogger("m31_d3")


# ------------------------------------------------------------------ data types


@dataclass(frozen=True)
class LineRef:
    """One selected 4TU line and the bundles that chose it."""

    activity_id: str
    path: str
    site: int
    n_traces: int
    dt_ns: float
    eps_r_used: float
    bundles: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.activity_id}/{self.path}"


@dataclass(frozen=True)
class Apex:
    blind_id: str
    activity_id: str
    path: str
    trace: int
    sample: float


@dataclass
class FoldResult:
    """Merged boxes of one fold on one line, plus pre-merge tile boxes."""

    tile_boxes: list[LineBox]
    merged: list[tuple[LineBox, GridBox]]


@dataclass
class Tables:
    """Rows of every output csv, filled line by line."""

    boxes: list[dict] = field(default_factory=list)
    r2_pairs: list[dict] = field(default_factory=list)
    band_pairs: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    apexes: list[dict] = field(default_factory=list)
    line_counts: list[dict] = field(default_factory=list)


# ------------------------------------------------------------ line selection


def _activity_key(activity_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in activity_id.split("."))


def _natural_key(text: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _dropped_duplicates(
    lines: Sequence[dict[str, str]], duplicates: Sequence[dict[str, str]]
) -> set[tuple[str, str]]:
    """Copies removed by keep_one (spec_eval_4tu_v1 §4-5-4)."""
    dropped = set()
    for row in duplicates:
        if row["scope"] == WITHIN_ACTIVITY:
            higher = max(row["path_a"], row["path_b"], key=_natural_key)
            dropped.add((row["activity_a"], higher))
    unique_counts = Counter(
        row["activity_id"]
        for row in lines
        if (row["activity_id"], row["path"]) not in dropped
    )
    shared: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in duplicates:
        if row["scope"] != ACROSS_ACTIVITIES:
            continue
        files = sorted(
            [(row["activity_a"], row["path_a"]), (row["activity_b"], row["path_b"])],
            key=lambda item: _activity_key(item[0]),
        )
        shared.setdefault((files[0][0], files[1][0]), []).append(
            (files[0][1], files[1][1])
        )
    for (low, high), paths in shared.items():
        n_shared, n_low, n_high = len(paths), unique_counts[low], unique_counts[high]
        drop_low = n_shared == n_low < n_high  # high is a strict superset
        side, activity = (0, low) if drop_low else (1, high)
        dropped |= {(activity, pair[side]) for pair in paths}
    return dropped


def _read_apexes() -> list[Apex]:
    apexes = []
    for row in _read_csv(APEX_REVIEW_CSV):
        if row["type"] != "T":
            continue
        apexes.append(
            Apex(
                row["blind_id"],
                row["activity_id"],
                row["path"],
                int(row["trace"]),
                float(row["sample"]),
            )
        )
    return apexes


def _first_lines_by_site(rows: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    """Bundle S: per site the activity with most lines, its first two paths."""
    chosen = []
    for site in sorted({int(row["site"]) for row in rows}):
        site_rows = [row for row in rows if int(row["site"]) == site]
        per_activity = Counter(row["activity_id"] for row in site_rows)
        activity = min(per_activity, key=lambda a: (-per_activity[a], _activity_key(a)))
        paths = sorted(
            (row["path"] for row in site_rows if row["activity_id"] == activity),
            key=_natural_key,
        )
        chosen += [(activity, path) for path in paths[:S_LINES_PER_SITE]]
    return chosen


def _special_lines(rows: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    """Bundle X: dt 0.117 lines and the first line of each listed eps_r."""
    ordered = sorted(
        rows, key=lambda r: (_activity_key(r["activity_id"]), _natural_key(r["path"]))
    )
    chosen = [
        (r["activity_id"], r["path"]) for r in ordered if float(r["dt_ns"]) == X_DT_NS
    ]
    for eps in X_EPS_VALUES:
        first = next(r for r in ordered if float(r["eps_r_used"]) == eps)
        chosen.append((first["activity_id"], first["path"]))
    return chosen


def select_lines(apexes: Sequence[Apex]) -> tuple[list[LineRef], list[str]]:
    """Bundles S, K, L, X (§13-3), duplicates removed; notes for the report."""
    all_rows = _read_csv(LINES_CSV)
    dropped = _dropped_duplicates(all_rows, _read_csv(DUPLICATES_CSV))
    rows = [r for r in all_rows if (r["activity_id"], r["path"]) not in dropped]
    by_key = {(r["activity_id"], r["path"]): r for r in all_rows}
    notes = [f"duplicate copies removed before selection: {len(dropped)}"]

    bundles: dict[str, list[tuple[str, str]]] = {
        "S": _first_lines_by_site(rows),
        "K": sorted(
            {(a.activity_id, a.path) for a in apexes},
            key=lambda k: (_activity_key(k[0]), _natural_key(k[1])),
        ),
        "L": [
            (r["activity_id"], r["path"])
            for r in sorted(rows, key=lambda r: -int(r["n_traces"]))[:L_LONGEST]
        ],
        "X": _special_lines(rows),
    }
    for key in bundles["K"]:
        if key in dropped:
            notes.append(
                f"K line {key[0]}/{key[1]} is a duplicate copy; kept (review label)"
            )
    membership: dict[tuple[str, str], list[str]] = {}
    for bundle, keys in bundles.items():
        for key in keys:
            membership.setdefault(key, []).append(bundle)
    refs = []
    for key in sorted(
        membership, key=lambda k: (_activity_key(k[0]), _natural_key(k[1]))
    ):
        row = by_key[key]
        refs.append(
            LineRef(
                key[0],
                key[1],
                int(row["site"]),
                int(row["n_traces"]),
                float(row["dt_ns"]),
                float(row["eps_r_used"]),
                tuple(membership[key]),
            )
        )
    notes += [f"bundle {b}: {len(k)} lines" for b, k in bundles.items()]
    return refs, notes


def write_csv(
    path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None
) -> None:
    names = list(fieldnames or (rows[0].keys() if rows else []))
    if not names:
        path.write_bytes(b"")  # no rows and no known columns: an empty file
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- measuring


def _grid_iou(a: GridBox, b: GridBox) -> float:
    width = min(a.i1, b.i1) - max(a.i0, b.i0) + 1
    height = min(a.k1, b.k1) - max(a.k0, b.k0) + 1
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / (a.n_traces * a.n_samples + b.n_traces * b.n_samples - inter)


def _r2_pair_rows(
    line: LineRef, fold: int, boxes: Sequence[LineBox], image: DetectionImage
):
    """Every R2-duplicate hyperbola pair from different tiles, with V1/V2 values."""
    config = DEFAULT_MERGE_CONFIG
    hyperbolas = [b for b in boxes if b.class_id == DetectionClass.PIPE_HYPERBOLA]
    rows = []
    for n, a in enumerate(hyperbolas):
        for b in hyperbolas[n + 1 :]:
            if a.tile_index == b.tile_index or not _is_r2_duplicate(a, b, config):
                continue
            top_p = abs(a.box.y1 - b.box.y1) / image.period_l1_px
            radius_px = _fresnel_radius_px(a, b, image)
            limit_px = max(radius_px, config.veto_min_center_px)
            gap_px = abs(a.center_x - b.center_x)
            either_cut = a.tile_truncated or b.tile_truncated
            v1 = top_p > config.veto_period_factor
            rows.append(
                {
                    "line": line.name,
                    "fold": fold,
                    "tile_a": a.tile_index,
                    "tile_b": b.tile_index,
                    "top_diff_p": round(top_p, 4),
                    "center_gap_px": round(gap_px, 3),
                    "fresnel_px": round(radius_px, 3),
                    "gap_over_limit": round(gap_px / limit_px, 4),
                    "either_truncated": either_cut,
                    "v1_veto": v1,
                    "v2_veto": (not v1) and (not either_cut) and gap_px > limit_px,
                    "v1_near": V1_NEAR_RANGE_P[0] <= top_p <= V1_NEAR_RANGE_P[1],
                    "x_a": round(a.center_x, 2),
                    "x_b": round(b.center_x, 2),
                }
            )
    return rows


def _band_pair_rows(line: LineRef, fold: int, merged, image: DetectionImage):
    """Neighbouring-tile band fragments: top difference and geometric candidacy."""
    bands = [(lb, g) for lb, g in merged if lb.class_id == DetectionClass.PIPE_BAND]
    grid_boxes = [g for _, g in bands]
    candidates = set(candidate_pairs(grid_boxes, image.dx_m, DEFAULT_BAND_CONFIG))
    rows = []
    for i, (a, ga) in enumerate(bands):
        for j in range(i + 1, len(bands)):
            b, gb = bands[j]
            if abs(a.tile_index - b.tile_index) != 1:
                continue
            gap_traces = max(gb.i0 - ga.i1 - 1, ga.i0 - gb.i1 - 1)
            smaller = min(ga.n_samples, gb.n_samples)
            overlap = max(0, min(ga.k1, gb.k1) - max(ga.k0, gb.k0) + 1)
            top_px = abs(a.box.y1 - b.box.y1)
            rows.append(
                {
                    "line": line.name,
                    "fold": fold,
                    "tile_a": a.tile_index,
                    "tile_b": b.tile_index,
                    "top_diff_px": round(top_px, 3),
                    "top_diff_p": round(top_px / image.period_l1_px, 4),
                    "smaller_height_px": round(
                        min(a.box.y2 - a.box.y1, b.box.y2 - b.box.y1), 3
                    ),
                    "time_overlap_frac": round(overlap / smaller, 4),
                    "gap_traces": gap_traces,
                    "is_candidate": (i, j) in candidates,
                }
            )
    return rows


def _apex_hit(apex: Apex, merged, period_samples: float) -> bool:
    margin = APEX_PERIOD_MARGIN * period_samples
    for line_box, g in merged:
        if line_box.class_id != DetectionClass.PIPE_HYPERBOLA:
            continue
        inside_traces = (
            g.i0 - APEX_TRACE_MARGIN <= apex.trace <= g.i1 + APEX_TRACE_MARGIN
        )
        if inside_traces and g.k0 - margin <= apex.sample <= g.k1 + margin:
            return True
    return False


def _n_agree(box: LineBox, grid: GridBox, results: dict[int, FoldResult]) -> int:
    """Folds (including fold 0) with a same-class L0 box of IoU >= 0.5."""
    count = 1
    for fold, result in results.items():
        if fold == MAIN_FOLD:
            continue
        if any(
            other.class_id == box.class_id and _grid_iou(grid, g) >= AGREE_IOU
            for other, g in result.merged
        ):
            count += 1
    return count


def _flag_row(line: LineRef, image: DetectionImage) -> dict:
    return {
        "line": line.name,
        "bundles": "+".join(line.bundles),
        "eps_r_used": line.eps_r_used,
        "period_samples": round(image.period_samples, 4),
        "period_source": str(image.period_source),
        "compression": round(image.compression, 4),
        "width_px": image.width_px,
        "content_height_px": image.content_height_px,
        "n_tiles": len(image.tiles),
        "overlap_px": image.overlap_px,
        "overlap_phys_px": image.overlap_phys_px,
        "overlap_period_capped": "overlap_period_capped" in image.flags,
        "period_compressed": "period_compressed" in image.flags,
        "scale_out_of_range": "scale_out_of_range" in image.flags,
        "flags": ";".join(image.flags),
    }


def _box_rows(line: LineRef, image: DetectionImage, results: dict[int, FoldResult]):
    rows = []
    p_star = image.p_target_px
    for fold, result in results.items():
        for line_box, g in result.merged:
            width = line_box.box.x2 - line_box.box.x1
            height = line_box.box.y2 - line_box.box.y1
            rows.append(
                {
                    "line": line.name,
                    "fold": fold,
                    "class_id": line_box.class_id,
                    "class_name": line_box.class_name,
                    "score": round(line_box.score, 4),
                    "x1": round(line_box.box.x1, 2),
                    "y1": round(line_box.box.y1, 2),
                    "x2": round(line_box.box.x2, 2),
                    "y2": round(line_box.box.y2, 2),
                    "width_px": round(width, 2),
                    "height_px": round(height, 2),
                    "width_p": round(width / p_star, 3),
                    "height_p": round(height / p_star, 3),
                    "i0": g.i0,
                    "i1": g.i1,
                    "k0": g.k0,
                    "k1": g.k1,
                    "tile_index": line_box.tile_index,
                    "n_tiles_seen": line_box.n_tiles_seen,
                    "tile_truncated": line_box.tile_truncated,
                    "tile_clipped": line_box.tile_clipped,
                    "n_agree": _n_agree(line_box, g, results)
                    if fold == MAIN_FOLD
                    else "",
                }
            )
    return rows


def _count_row(line: LineRef, fold: int, result: FoldResult) -> dict:
    before = Counter(b.class_name for b in result.tile_boxes)
    after = Counter(lb.class_name for lb, _ in result.merged)
    row = {"line": line.name, "fold": fold}
    for member in DetectionClass:
        row[f"{member.label}_tiles"] = before.get(member.label, 0)
        row[f"{member.label}_merged"] = after.get(member.label, 0)
    row["tile_boxes_truncated"] = sum(b.tile_truncated for b in result.tile_boxes)
    row["tile_boxes"] = len(result.tile_boxes)
    row["tile_seam_box"] = sum(lb.tile_truncated for lb, _ in result.merged)
    return row


# ------------------------------------------------------------------ gallery


def _draw_gallery(
    out_dir: Path,
    line: LineRef,
    image: DetectionImage,
    result: FoldResult,
    vetoed: Sequence[dict],
    apexes: Sequence[Apex],
) -> int:
    """Fold-0 line L1 image with boxes, seams, vetoes and apexes, 1,024 px chunks."""
    canvas = cv2.cvtColor(image.line_image, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(
        canvas,
        None,
        fx=GALLERY_SCALE,
        fy=GALLERY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    s = GALLERY_SCALE
    height = canvas.shape[0]
    for tile in image.tiles:
        for seam in tile.seams_px:
            cv2.line(canvas, (seam * s, 0), (seam * s, height - 1), SEAM_COLOR_BGR, 1)
    for line_box, _ in result.merged:
        color = CLASS_COLORS_BGR[DetectionClass(line_box.class_id)]
        thickness = TRUNCATED_THICKNESS if line_box.tile_truncated else 1
        top_left = (round(line_box.box.x1 * s), round(line_box.box.y1 * s))
        bottom_right = (round(line_box.box.x2 * s), round(line_box.box.y2 * s))
        cv2.rectangle(canvas, top_left, bottom_right, color, thickness)
    for pair in vetoed:
        y = 4 * s
        cv2.line(
            canvas,
            (round(pair["x_a"] * s), y),
            (round(pair["x_b"] * s), y),
            VETO_COLOR_BGR,
            2,
        )
    grid = image.grid_map
    for apex in apexes:
        u = round(apex.trace / grid.traces_per_px * s)
        v = round(apex.sample / grid.samples_per_px * s)
        cv2.drawMarker(canvas, (u, v), APEX_COLOR_BGR, cv2.MARKER_CROSS, 10, 1)
    chunk = GALLERY_CHUNK_PX * s
    n_files = 0
    stem = line.name.replace("/", "_")
    for start in range(0, canvas.shape[1], chunk):
        path = out_dir / "gallery" / f"{stem}_x{start // s:05d}.png"
        if not cv2.imwrite(str(path), canvas[:, start : start + chunk]):
            raise OSError(f"could not write {path}")
        n_files += 1
    return n_files


# --------------------------------------------------------------------- runs


def run_line(
    line: LineRef,
    root: Path,
    detectors: dict,
    tables: Tables,
    apexes: Sequence[Apex],
    out_dir: Path,
    gallery: bool,
) -> None:
    radargram = load_fourtu_radargram(root, line.activity_id, line.path)
    image = make_detection_image(radargram)
    results: dict[int, FoldResult] = {}
    for fold, detector in detectors.items():
        tile_boxes = detect_tiles(image, detector, DEFAULT_MERGE_CONFIG.edge_px)
        merged = line_boxes_to_grid(merge_line_boxes(tile_boxes, image), image)
        results[fold] = FoldResult(tile_boxes, merged)
        tables.r2_pairs += _r2_pair_rows(line, fold, tile_boxes, image)
        tables.band_pairs += _band_pair_rows(line, fold, merged, image)
        tables.line_counts.append(_count_row(line, fold, results[fold]))
    tables.flags.append(_flag_row(line, image))
    tables.boxes += _box_rows(line, image, results)
    line_apexes = [
        a for a in apexes if (a.activity_id, a.path) == (line.activity_id, line.path)
    ]
    for apex in line_apexes:
        hits = {
            fold: _apex_hit(apex, r.merged, image.period_samples)
            for fold, r in results.items()
        }
        tables.apexes.append(
            {
                "blind_id": apex.blind_id,
                "line": line.name,
                "trace": apex.trace,
                "sample": apex.sample,
                "hit_fold0": hits.get(MAIN_FOLD, ""),
                "folds_hit": sum(hits.values()),
                "folds_run": len(hits),
            }
        )
    if gallery and MAIN_FOLD in results:
        vetoed = [
            row
            for row in tables.r2_pairs
            if row["line"] == line.name
            and row["fold"] == MAIN_FOLD
            and (row["v1_veto"] or row["v2_veto"])
        ]
        _draw_gallery(out_dir, line, image, results[MAIN_FOLD], vetoed, line_apexes)
    LOGGER.info(
        "%s: %d tiles, merged per fold %s",
        line.name,
        len(image.tiles),
        {f: len(r.merged) for f, r in results.items()},
    )


def _distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0}
    q = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "n": int(array.size),
        "p5": q[0],
        "p25": q[1],
        "p50": q[2],
        "p75": q[3],
        "p95": q[4],
        "max": float(array.max()),
    }


def summarise(tables: Tables, folds: Sequence[int]) -> dict:
    main_boxes = [r for r in tables.boxes if r["fold"] == MAIN_FOLD]
    hyperbolas = [
        r for r in main_boxes if r["class_id"] == DetectionClass.PIPE_HYPERBOLA
    ]
    main_pairs = [r for r in tables.r2_pairs if r["fold"] == MAIN_FOLD]
    main_counts = [r for r in tables.line_counts if r["fold"] == MAIN_FOLD]
    band_pairs = [r for r in tables.band_pairs if r["fold"] == MAIN_FOLD]
    return {
        "fold0_boxes_by_class": dict(Counter(r["class_name"] for r in main_boxes)),
        "fold0_tile_boxes": sum(r["tile_boxes"] for r in main_counts),
        "fold0_tile_boxes_truncated": sum(
            r["tile_boxes_truncated"] for r in main_counts
        ),
        "fold0_tile_seam_box": sum(r["tile_seam_box"] for r in main_counts),
        "fold0_hyperbola_width_px": _distribution(r["width_px"] for r in hyperbolas),
        "fold0_hyperbola_height_px": _distribution(r["height_px"] for r in hyperbolas),
        "fold0_n_tiles_seen": dict(Counter(r["n_tiles_seen"] for r in main_boxes)),
        "fold0_n_agree": dict(Counter(r["n_agree"] for r in main_boxes)),
        "fold0_r2_pairs": len(main_pairs),
        "fold0_v1_vetoes": sum(r["v1_veto"] for r in main_pairs),
        "fold0_v2_vetoes": sum(r["v2_veto"] for r in main_pairs),
        "fold0_v1_near_pairs": sum(r["v1_near"] for r in main_pairs),
        "fold0_v1_veto_top_diff_p": _distribution(
            r["top_diff_p"] for r in main_pairs if r["v1_veto"]
        ),
        "fold0_band_neighbour_pairs": len(band_pairs),
        "fold0_band_pairs_not_candidate": sum(
            not r["is_candidate"] for r in band_pairs
        ),
        "apexes": len(tables.apexes),
        "apexes_hit_fold0": sum(r["hit_fold0"] is True for r in tables.apexes),
        "apex_folds_hit": dict(Counter(r["folds_hit"] for r in tables.apexes)),
        "flag_counts": {
            flag: sum(bool(r[flag]) for r in tables.flags)
            for flag in (
                "overlap_period_capped",
                "period_compressed",
                "scale_out_of_range",
            )
        },
        "folds_run": list(folds),
    }


def run_identity(args: argparse.Namespace, weights: dict[int, Path]) -> dict:
    return {
        "script": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "argv": sys.argv[1:],
        "started": local_timestamp(),
        "git": git_state(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "packages": package_versions(
            ["numpy", "opencv-python", "torch", "ultralytics"]
        ),
        "device": args.device,
        "confidence": CONFIDENCE,
        "image_config": asdict(DEFAULT_DETECTION_IMAGE_CONFIG),
        "merge_config": asdict(DEFAULT_MERGE_CONFIG),
        "weights": {
            str(f): {
                "path": p.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(p),
            }
            for f, p in weights.items()
        },
    }


def _prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; refusing to overwrite")
    (out_dir / "gallery").mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0", help="torch device, e.g. cpu")
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    parser.add_argument(
        "--limit-lines",
        type=int,
        default=None,
        help="smoke test: only the first N selected lines",
    )
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="write lines.csv and stop (no models loaded)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    try:
        _prepare_out_dir(out_dir)
    except FileExistsError as error:
        LOGGER.error("%s", error)
        return 2

    apexes = _read_apexes()
    lines, notes = select_lines(apexes)
    lines_csv = out_dir / "lines.csv"
    write_csv(
        lines_csv,
        [
            {
                "activity_id": r.activity_id,
                "path": r.path,
                "site": r.site,
                "n_traces": r.n_traces,
                "dt_ns": r.dt_ns,
                "eps_r_used": r.eps_r_used,
                "bundles": "+".join(r.bundles),
            }
            for r in lines
        ],
    )
    lines_sha256 = sha256_file(lines_csv)
    LOGGER.info("selected %d lines, lines.csv sha256 %s", len(lines), lines_sha256)
    if args.select_only:
        return 0

    weights = {f: PROJECT_ROOT / WEIGHTS_PATTERN.format(fold=f) for f in args.folds}
    identity = run_identity(args, weights)
    detectors = {
        f: load_yolo_detector(p, confidence_threshold=CONFIDENCE, device=args.device)
        for f, p in weights.items()
    }
    root = load_dataset_paths().root(DatasetName.FOURTU)
    run_lines = lines if args.limit_lines is None else lines[: args.limit_lines]
    tables = Tables()
    for line in run_lines:
        run_line(line, root, detectors, tables, apexes, out_dir, not args.no_gallery)

    for name, rows in (
        ("boxes", tables.boxes),
        ("r2_pairs", tables.r2_pairs),
        ("band_pairs", tables.band_pairs),
        ("line_flags", tables.flags),
        ("known_apexes", tables.apexes),
        ("line_counts", tables.line_counts),
    ):
        write_csv(out_dir / f"{name}.csv", rows)
    summary = {
        "run_identity": {**identity, "finished": local_timestamp()},
        "lines_csv_sha256": lines_sha256,
        "n_lines_selected": len(lines),
        "n_lines_run": len(run_lines),
        "selection_notes": notes,
        "measurements": summarise(tables, args.folds),
    }
    text = json.dumps(summary, indent=1, ensure_ascii=False, default=float)
    (out_dir / "d3_summary.json").write_text(
        text + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
