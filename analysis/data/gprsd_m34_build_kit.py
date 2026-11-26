"""Build the GPR-SD M3-4 pipe-labelling kit (windows, label-view images, Label Studio tasks).

Plan: docs/processing/m36_gprsd_plan.md section 5. Inputs are read-only.
Outputs go to data/labels/gprsd_m34/ (git-ignored) plus a public manifest with sha256/counts only.

Run: conda run -n gpr-dev python -I analysis/data/gprsd_m34_build_kit.py
No signal-processing output (M3-6) and no detector is used anywhere in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20261126
HALF = 300  # window half width in traces (plan 5-1)
SCALE_X = 2  # px per trace
SCALE_Y = 2  # px per sample
N_RELABEL = 6
MARK_TYPE = "pipeline"

REPO = Path(__file__).resolve().parents[2]
PKG = Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_marks(pkg: Path) -> list[dict]:
    with open(pkg / "01_메타데이터" / "marks.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["type"] == MARK_TYPE]


def read_index(pkg: Path) -> dict[str, dict]:
    with open(pkg / "01_메타데이터" / "radargrams_index.csv", encoding="utf-8-sig", newline="") as f:
        return {r["file_id"]: r for r in csv.DictReader(f)}


def build_windows(marks: list[dict], index: dict[str, dict]) -> list[dict]:
    """Per line: [mark-HALF, mark+HALF] clipped to the line, overlapping windows merged."""
    out = []
    for file_id, meta in sorted(index.items()):
        road = meta["road"]
        n_tr = int(meta["n_traces"])
        road_marks = sorted(
            ((int(m["location_trace"]), m["xlsx_row"]) for m in marks if m["road"] == road)
        )
        spans: list[list] = []
        for loc, row in road_marks:
            a, b = max(0, loc - HALF), min(n_tr - 1, loc + HALF)
            if spans and a <= spans[-1][1]:  # overlap (inclusive ranges)
                spans[-1][1] = max(spans[-1][1], b)
                spans[-1][2].append((loc, row))
            else:
                spans.append([a, b, [(loc, row)]])
        for a, b, ms in spans:
            out.append(
                {
                    "window_id": f"{file_id}_t{a:05d}",
                    "file_id": file_id,
                    "road": road,
                    "line_no": int(meta["line_no"]),
                    "window_trace0": a,
                    "window_trace1": b,
                    "n_traces_window": b - a + 1,
                    "sample0": 0,
                    "n_samples": int(meta["n_samples"]),
                    "mark_traces": "|".join(str(m[0]) for m in ms),
                    "mark_xlsx_rows": "|".join(str(m[1]) for m in ms),
                    "n_marks": len(ms),
                }
            )
    return out


def to_gray(win: np.ndarray, clip: float) -> np.ndarray:
    """Symmetric clip at +-clip -> uint8 (-clip -> 0, 0 -> 128, +clip -> 255)."""
    g = np.rint((np.clip(win.astype(np.float64) / clip, -1.0, 1.0) + 1.0) * 127.5)
    return g.astype(np.uint8)


def render_window(data: np.ndarray, clip: float, t0: int, t1: int) -> np.ndarray:
    win = np.asarray(data[:, t0 : t1 + 1])  # [samples, traces]
    g = to_gray(win, clip)
    return np.repeat(np.repeat(g, SCALE_Y, axis=0), SCALE_X, axis=1)


def write_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, format="PNG", compress_level=6, optimize=False)


def choose_relabel(windows: list[dict], order_main: list[int]) -> list[int]:
    """6 windows, all 4 roads present, <=2 per road, fixed seed. Returns window indices in relabel order."""
    rng = np.random.default_rng([SEED, 1])
    roads = sorted({w["road"] for w in windows})
    for _ in range(10000):
        pick = [int(i) for i in rng.choice(len(windows), size=N_RELABEL, replace=False)]
        cnt = {r: sum(windows[i]["road"] == r for i in pick) for r in roads}
        # parallel lines of one road share window positions: do not pick the same position twice
        uniq = len({(windows[i]["road"], windows[i]["window_trace0"]) for i in pick}) == N_RELABEL
        if min(cnt.values()) >= 1 and max(cnt.values()) <= 2 and uniq:
            break
    else:
        raise RuntimeError("relabel selection failed")
    # relabel presentation order: shuffled, and different from their relative order in the main bundle
    rank_main = {wi: k for k, wi in enumerate(order_main)}
    for _ in range(10000):
        perm = [int(i) for i in rng.permutation(pick)]
        main_rel = sorted(perm, key=lambda i: rank_main[i])
        if perm != main_rel:
            return perm
    raise RuntimeError("relabel order failed")


def task_json(w: dict, image_url: str, order: int, total: int, bundle: str) -> dict:
    # No "predictions" key at all (plan 5-3: no pre-annotation).
    return {
        "data": {
            "image": image_url,
            "order": order,
            "total": total,
            "bundle": bundle,
            "window_id": w["window_id"],
            "file_id": w["file_id"],
            "window_trace0": w["window_trace0"],
            "window_trace1": w["window_trace1"],
            "view_w_px": SCALE_X * w["n_traces_window"],
            "view_h_px": SCALE_Y * w["n_samples"],
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "labels" / "gprsd_m34")
    ap.add_argument("--manifest", type=Path, default=REPO / "docs" / "datasets" / "gprsd_m34_label_manifest.json")
    args = ap.parse_args()

    marks = read_marks(args.pkg)
    index = read_index(args.pkg)
    assert len(marks) == 17, f"expected 17 pipeline marks, got {len(marks)}"
    windows = build_windows(marks, index)
    n_lines = len(index)
    per_line = {fid: sum(w["file_id"] == fid for w in windows) for fid in index}
    print("lines", n_lines, "windows", len(windows), "per_line", sorted(set(per_line.values())))
    # each of the 17 marks is inside exactly one window per line
    for fid, meta in index.items():
        cov = sum(w["n_marks"] for w in windows if w["file_id"] == fid)
        exp = sum(1 for m in marks if m["road"] == meta["road"])
        assert cov == exp, (fid, cov, exp)
    # 17 marks x 3 lines = 51 mark-line pairs
    pairs = sum(w["n_marks"] for w in windows)
    assert pairs == 51, pairs

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # presentation order (shuffled across roads, fixed seed)
    rng = np.random.default_rng(SEED)
    order_main = [int(i) for i in rng.permutation(len(windows))]
    relabel_idx = choose_relabel(windows, order_main)

    # render
    clips: dict[str, float] = {}
    src_sha: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    for fid in sorted(index):
        npy = args.pkg / "05_radargrams_npy" / f"{fid}.npy"
        a = np.load(npy, mmap_mode="r")
        assert a.shape == (int(index[fid]["n_samples"]), int(index[fid]["n_traces"])), (fid, a.shape)
        clips[fid] = float(np.percentile(np.abs(np.asarray(a)), 99))
        src_sha[fid] = sha256_file(npy)
        arrays[fid] = a

    img_info: dict[str, dict] = {}
    for rank, wi in enumerate(order_main, start=1):
        w = windows[wi]
        arr = render_window(arrays[w["file_id"]], clips[w["file_id"]], w["window_trace0"], w["window_trace1"])
        assert arr.shape == (SCALE_Y * w["n_samples"], SCALE_X * w["n_traces_window"])
        rel = f"images/v{rank:02d}.png"
        write_png(arr, out / rel)
        w["order"] = rank
        w["image_file"] = rel
        w["image_sha256"] = sha256_file(out / rel)
        w["view_w_px"], w["view_h_px"] = arr.shape[1], arr.shape[0]
    # relabel copies (same pixels, new file name and new order)
    for rank, wi in enumerate(relabel_idx, start=1):
        w = windows[wi]
        arr = render_window(arrays[w["file_id"]], clips[w["file_id"]], w["window_trace0"], w["window_trace1"])
        rel = f"images_relabel/r{rank:02d}.png"
        write_png(arr, out / rel)
        w["relabel_order"] = rank
        w["relabel_image_file"] = rel
        w["relabel_image_sha256"] = sha256_file(out / rel)
        assert w["relabel_image_sha256"] == w["image_sha256"]  # identical pixels

    # windows.csv (canonical order: file_id, trace0)
    cols = [
        "window_id", "file_id", "road", "line_no", "window_trace0", "window_trace1", "n_traces_window",
        "sample0", "n_samples", "n_marks", "mark_traces", "mark_xlsx_rows", "view_w_px", "view_h_px",
        "order", "image_file", "image_sha256", "relabel", "relabel_order", "relabel_image_file", "window_complete",
    ]
    for w in windows:
        w["relabel"] = "yes" if "relabel_order" in w else "no"
        w.setdefault("relabel_order", "")
        w.setdefault("relabel_image_file", "")
        w["window_complete"] = ""  # filled by gprsd_m34_export_to_l0.py -> windows_status.csv
    with open(out / "windows.csv", "w", encoding="utf-8", newline="\n") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(windows)

    # render_config.json
    cfg = {
        "kit": "gprsd_m34",
        "seed": SEED,
        "scale_x_px_per_trace": SCALE_X,
        "scale_y_px_per_sample": SCALE_Y,
        "half_window_traces": HALF,
        "mapping": {
            "text": "pixel edge coordinates: trace_f = window_trace0 + x_px/scale_x ; sample_f = sample0 + y_px/scale_y ; "
                    "L0 cell i covers [i, i+1). GridBox = floor(t1+1e-6)..ceil(t2-1e-6)-1 (ADR-004 decision 1, 8).",
            "sample0": 0,
            "grid_rounding_tolerance": 1e-6,
        },
        "gray": {
            "clip_rule": "per-line p99 of |amp| over the whole line (float32 npy as stored, np.percentile linear)",
            "formula": "uint8 = rint((clip(a/clip, -1, 1) + 1) * 127.5); 0 -> 128",
            "colormap": "none (8-bit grey), no axes, no mark lines",
        },
        "clip_per_line": clips,
        "source_npy_sha256": src_sha,
        "windows": {
            w["window_id"]: {
                "window_trace0": w["window_trace0"],
                "window_trace1": w["window_trace1"],
                "image_file": w["image_file"],
                "image_sha256": w["image_sha256"],
                "view_w_px": w["view_w_px"],
                "view_h_px": w["view_h_px"],
            }
            for w in windows
        },
    }
    cfg_path = out / "render_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    cfg_sha = sha256_file(cfg_path)

    # Label Studio tasks
    base = "/data/local-files/?d=labels/gprsd_m34/"
    main_tasks = [
        task_json(windows[wi], base + windows[wi]["image_file"], r, len(windows), "main")
        for r, wi in enumerate(order_main, start=1)
    ]
    re_tasks = [
        task_json(windows[wi], base + windows[wi]["relabel_image_file"], r, len(relabel_idx), "relabel")
        for r, wi in enumerate(relabel_idx, start=1)
    ]
    for t in main_tasks + re_tasks:
        assert "predictions" not in t
    (out / "tasks_main.json").write_text(json.dumps(main_tasks, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "tasks_relabel.json").write_text(json.dumps(re_tasks, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # SHA256SUMS (LF)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{sha256_file(p)}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # public manifest: hashes and counts only (no data, no trace ranges)
    manifest = {
        "kit": "gprsd_m34",
        "built_by": "analysis/data/gprsd_m34_build_kit.py",
        "seed": SEED,
        "counts": {
            "pipeline_marks": len(marks),
            "lines": n_lines,
            "windows": len(windows),
            "windows_per_line": per_line,
            "mark_line_pairs_covered": pairs,
            "relabel_windows": len(relabel_idx),
            "relabel_per_road": {r: sum(windows[i]["road"] == r for i in relabel_idx) for r in sorted({w["road"] for w in windows})},
        },
        "render": {"scale_x": SCALE_X, "scale_y": SCALE_Y, "half_window_traces": HALF, "clip": "per-line p99 |amp|, 8-bit grey"},
        "sha256": {
            "windows.csv": sha256_file(out / "windows.csv"),
            "render_config.json": cfg_sha,
            "tasks_main.json": sha256_file(out / "tasks_main.json"),
            "tasks_relabel.json": sha256_file(out / "tasks_relabel.json"),
            "source_npy": src_sha,
            "images_main": {Path(w["image_file"]).name: w["image_sha256"] for w in sorted(windows, key=lambda x: x["order"])},
        },
        "note": "Data license unknown: source data, view images and labels are not published; only hashes and counts.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "ok", "windows": len(windows), "relabel": len(relabel_idx), "render_config_sha256": cfg_sha, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
