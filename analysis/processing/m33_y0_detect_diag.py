"""M3-3 D0 follow-up: why the line detector found almost no Y0 synthetic boxes.

Diagnostic only (the D0 verdict is closed). CPU, single process, BelowNormal.
No detector is run: only the r0 detection images of the gprMax models are
rebuilt (same calls as analysis/detection/m33_d0_gate.py ``synthetic_lines``)
and measured, and the recorded D0 boxes are re-read.

Outputs (analysis/processing/ and docs/processing/figures/m33_y0_diag/):
  m33_y0_detect_diag.json   per-model geometry / amplitude / texture metrics
  m33_y0_detect_diag.csv    one row per (model, condition)
  figures: y0_tiles.png, y1_05p_tiles.png, reference_tiles.png, size_scatter.png

Hypotheses measured
  H1 size: pipe-only envelope box at -6 / -20 dB in L1 periods vs AIHub
     tpipe label p5-p95 (width 7.79-14.67 P, height 5.22-8.35 P).
  H2 texture: content stats (R_h, rho_bt, std, sat) vs AIHub g1.0 reference
     and 4TU r0 (m32_texture_recipes.json); share of visible texture outside
     the pipe response.
  H3 imaging: tile fill (content area, right/bottom pad) vs training pad
     ranges; pre/post AGC floor relative to the pipe apex; clip level.
"""

from __future__ import annotations

import ctypes
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.paths import DatasetName, load_dataset_paths  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.io.synthetic import (  # noqa: E402
    load_synthetic,
    load_synthetic_background,
    load_synthetic_truth,
)
from gpr_detection.pipeline import content_rows_for, line_zero  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DEFAULT_DETECTION_IMAGE_CONFIG,
    _line_amplitudes,
    make_detection_image,
    resample_area,
)

SYN = ROOT / "data" / "synthetic_gprmax"
D0 = ROOT / "runs" / "eval" / "m33_d0"
OUT_JSON = ROOT / "analysis" / "processing" / "m33_y0_detect_diag.json"
OUT_CSV = ROOT / "analysis" / "processing" / "m33_y0_detect_diag.csv"
FIG = ROOT / "docs" / "processing" / "figures" / "m33_y0_diag"
TEXTURE_JSON = ROOT / "analysis" / "processing" / "m32_texture_recipes.json"
AIHUB_REF_CSV = ROOT / "analysis" / "processing" / "m32_stageA_aihub_ref.csv"
LINES_4TU_CSV = ROOT / "analysis" / "processing" / "m32_stageA_lines_4tu.csv"
AIHUB_IMAGES = ROOT / "data" / "aihub_yolo_v1" / "images"
AIHUB_LABELS = ROOT / "data" / "aihub_yolo_v1" / "labels"

MODELS = tuple(f"M{i:02d}" for i in range(1, 21))
CUTS = (0.5, 1.0, 2.0, 3.0)
AIHUB_W = (7.79, 14.67)
AIHUB_H = (5.22, 8.35)
TRAIN_SIDE_PAD_MAX = 96
TRAIN_BOTTOM_PAD_MAX = 128
TRAIN_MIN_SCALE = 0.55
SEED = 20261115
CFG = DEFAULT_DETECTION_IMAGE_CONFIG


def below_normal() -> None:
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def stats(q: np.ndarray) -> dict[str, float | None]:
    """Same definition as m32_texture_recipes.stats (stage A)."""
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


def envelope(section: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(np.asarray(section, dtype=np.float64), axis=0))


def seed_box(mask: np.ndarray, seed: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """(col0, col1, row0, row1) of the component of ``mask`` holding ``seed``."""
    lab, _ = ndimage.label(mask)
    v = lab[seed]
    if v == 0:
        return None
    rows, cols = np.nonzero(lab == v)
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def pipe_only(radargram: Radargram, model: str) -> np.ndarray:
    """Full minus the no-pipe run (middle bg trace; the soil is x-invariant)."""
    bg = load_synthetic_background(model, root=SYN)[: radargram.n_samples]
    return np.asarray(radargram.data, dtype=np.float64) - bg[:, 1:2].astype(np.float64)


def bg_x_variation(model: str) -> float:
    bg = load_synthetic_background(model, root=SYN).astype(np.float64)
    ref = np.max(np.abs(bg[:, 1]))
    return float(np.max(np.abs(bg[:, [0, 2]] - bg[:, 1:2])) / ref)


def db(x: float) -> float | None:
    return None if not x > 0 else float(20 * np.log10(x))


def line_metrics(model: str, cond: str, rg: Radargram, apex_trace: float, apex_sample: float,
                 period_override: float | None) -> tuple[dict, object]:
    img = make_detection_image(rg, CFG, period_samples_override=period_override)
    amps = _line_amplitudes(rg, CFG, period_override)
    branch = prepare_fitting_branch(rg, CFG.period_bg_window)
    content = rows = content_rows_for(img, line_zero(rg))
    H, W = img.content_height_px, img.width_px
    spp, tpp = img.grid_map.samples_per_px, img.grid_map.traces_per_px
    p_l1 = img.period_samples / spp

    # pipe-only response through the linear steps (dc, dewow, band-pass)
    po = replace(rg, data=pipe_only(rg, model).astype(np.float32))
    po_f = prepare_fitting_branch(po, CFG.period_bg_window).filtered
    env = envelope(po_f)
    env_l1 = resample_area(env, (H, W))
    ar, ac = min(H - 1, int(apex_sample / spp)), min(W - 1, int(apex_trace / tpp))
    # apex peak in L1 near the truth apex (+-1 P)
    r0, r1 = max(0, ar - int(p_l1)), min(H, ar + int(p_l1) + 1)
    c0, c1 = max(0, ac - 3), min(W, ac + 4)
    sub = env_l1[r0:r1, c0:c1]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    seed = (r0 + k[0], c0 + k[1])
    peak = float(env_l1[seed])
    boxes = {}
    for level in (-6, -10, -20):
        b = seed_box(env_l1 >= peak * 10 ** (level / 20), seed)
        if b is None:
            boxes[level] = None
            continue
        boxes[level] = {
            "box_px": b,
            "w_P": (b[1] - b[0] + 1) / p_l1,
            "h_P": (b[3] - b[2] + 1) / p_l1,
            "aperture_limited": bool(b[0] == 0 or b[1] == W - 1),
            "bottom_limited": bool(b[3] == H - 1),
        }

    # amplitude floor relative to the pipe apex, pre / post AGC (L0)
    env_po_mask = env >= env.max() * 1e-2  # -40 dB pipe support
    zero_k = None if rows.zero_px is None else rows.zero_px * spp
    below = np.zeros_like(env_po_mask)
    start = int((zero_k or 0) + 2 * img.period_samples)
    below[start:, :] = True
    floor_region = below & ~ndimage.binary_dilation(env_po_mask, iterations=5)
    pre = np.asarray(branch.background_removed, dtype=np.float64)
    post = np.asarray(amps.amplitudes, dtype=np.float64)
    env_pre, env_post = envelope(pre), envelope(post)
    apex_pre = float(env_pre[max(0, int(apex_sample) - 20): int(apex_sample) + 21,
                             int(apex_trace)].max())
    apex_post = float(env_post[max(0, int(apex_sample) - 20): int(apex_sample) + 21,
                               int(apex_trace)].max())
    rms = lambda a, m: float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else 0.0  # noqa: E731
    floor_pre = rms(pre, floor_region)
    floor_post = rms(post, floor_region)
    clip = float(np.percentile(np.abs(post), CFG.clip_percentile))

    # grey image (content) texture outside the pipe response and the top band
    gray = img.line_image[:H].astype(np.float64)
    pipe_l1 = env_l1 >= peak * 0.1
    pipe_l1 = ndimage.binary_dilation(pipe_l1, iterations=2)
    zero_px = rows.zero_px or 0.0
    top_rows = int(np.ceil(zero_px + 2 * p_l1))
    off = np.ones_like(pipe_l1)
    off[:top_rows] = False
    off &= ~pipe_l1
    dev = np.abs(gray - 127.5)
    vis_off = float(np.mean(dev[off] > 16)) if off.any() else None
    vis_pipe = float(np.mean(dev[pipe_l1] > 16)) if pipe_l1.any() else None
    top_dev = float(dev[:top_rows].mean())
    apex_gray_dev = float(dev[max(0, seed[0] - 2): seed[0] + 3, max(0, seed[1] - 2): seed[1] + 3].max())

    out = {
        "model": model,
        "condition": cond,
        "n_samples": rg.n_samples,
        "period_samples": img.period_samples,
        "period_l1_px": p_l1,
        "compression": img.compression,
        "content_w_px": W,
        "content_h_px": H,
        "n_tiles": len(img.tiles),
        "pad_right_px": max(0, CFG.tile_px - W),
        "pad_bottom_px": CFG.tile_px - H,
        "content_area_frac": W * H / CFG.tile_px**2 if W <= CFG.tile_px else H / CFG.tile_px,
        "zero_px": rows.zero_px,
        "y_bot_px": rows.y_bot_px,
        "apex_row_px": seed[0],
        "apex_col_px": seed[1],
        "box": {str(k): v for k, v in boxes.items()},
        "pipe_frac_of_content": float(np.mean(env_l1 >= peak * 0.1)),
        "floor_pre_db": db(floor_pre / apex_pre) if apex_pre > 0 else None,
        "floor_post_db": db(floor_post / apex_post) if apex_post > 0 else None,
        "agc_floor_lift_db": (db(floor_post / apex_post) - db(floor_pre / apex_pre))
        if floor_pre > 0 and apex_pre > 0 else None,
        "apex_post_over_clip": apex_post / clip if clip > 0 else None,
        "visible_off_pipe_frac": vis_off,
        "visible_pipe_frac": vis_pipe,
        "top_band_mean_dev": top_dev,
        "apex_gray_dev": apex_gray_dev,
        "stats": stats(gray),
        "flags": list(img.flags),
    }
    return out, img


def synthetic_lines(model: str) -> list[tuple[dict, object]]:
    rg = load_synthetic(model, root=SYN)
    truth = load_synthetic_truth(model, root=SYN)
    origin = float(rg.metadata.get("time_of_sample0_ns", 0.0))
    apex_trace = float(truth.apex_trace_index)
    apex_sample = (truth.apex_envelope_peak_ns - origin) / rg.dt_ns
    out = []
    m0, img0 = line_metrics(model, "Y0", rg, apex_trace, apex_sample, None)
    m0["bg_x_variation"] = bg_x_variation(model)
    out.append((m0, img0))
    period = img0.period_samples
    for f in CUTS:
        n_keep = min(rg.n_samples, int(np.floor(apex_sample + f * period)) + 1)
        cut = replace(rg, data=np.ascontiguousarray(rg.data[:n_keep]))
        out.append(line_metrics(model, f"Y1_{f:g}P", cut, apex_trace, apex_sample, period))
    return out


def d0_boxes() -> list[dict]:
    rows = []
    for k in range(5):
        with open(D0 / f"fold{k}" / "synthetic_boxes.csv", newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


# ------------------------------------------------------------------ figures
def tile_grid(tiles, titles, boxes_by_idx, path, ncol=5, apex=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(tiles)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.75))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(tiles[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(titles[i], fontsize=8, color="#0b0b0b")
        for x1, y1, x2, y2, label, color in boxes_by_idx.get(i, []):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec=color))
            ax.text(x1, max(0, y1) - 2, label, fontsize=6, color=color, va="bottom")
        if apex is not None and apex[i] is not None:
            ax.plot(apex[i][1], apex[i][0], marker="+", color="#eb6834", ms=8, mew=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def size_scatter(rows: list[dict], hits: set[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.add_patch(Rectangle((AIHUB_W[0], AIHUB_H[0]), AIHUB_W[1] - AIHUB_W[0],
                           AIHUB_H[1] - AIHUB_H[0], fc="#cde2fb", ec="#2a78d6", lw=1,
                           label="AIHub tpipe label p5-p95"))
    for level, marker, color in (("-6", "o", "#2a78d6"), ("-20", "s", "#eb6834")):
        xs = [r["box"][level]["w_P"] for r in rows if r["box"][level]]
        ys = [r["box"][level]["h_P"] for r in rows if r["box"][level]]
        ax.scatter(xs, ys, s=36, marker=marker, facecolor="none", edgecolor=color, lw=1.4,
                   label=f"synthetic Y0 pipe envelope {level} dB")
        for r in rows:
            b = r["box"][level]
            if b and r["model"] in hits:
                ax.annotate(r["model"], (b["w_P"], b["h_P"]), fontsize=7, color="#52514e",
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("box width [L1 periods P]")
    ax.set_ylabel("box height [L1 periods P]")
    ax.set_title("Y0 pipe-only envelope box (before AGC) vs AIHub labels\n"
                 "(named: D0 hit models; after AGC every -6 dB box spans the whole line)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reference_tiles(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    with open(AIHUB_REF_CSV, newline="", encoding="utf-8") as fh:
        ref = [r for r in csv.DictReader(fh) if r["class"] == "pipe_hyperbola" and r["g1.0"]]
    pick = [ref[i] for i in sorted(rng.choice(len(ref), 5, replace=False))]
    tiles, titles, boxes = [], [], {}
    for j, r in enumerate(pick):
        q = np.asarray(Image.open(AIHUB_IMAGES / r["g1.0"]).convert("L"))
        tiles.append(q)
        titles.append("AIHub " + r["g1.0"][:14])
        lab = AIHUB_LABELS / (Path(r["g1.0"]).stem + ".txt")
        bl = []
        for line in lab.read_text().splitlines():
            c, xc, yc, w, h = map(float, line.split())
            if int(c) == 0:
                bl.append(((xc - w / 2) * 256, (yc - h / 2) * 256, (xc + w / 2) * 256,
                           (yc + h / 2) * 256, "label", "#1baf7a"))
        boxes[j] = bl
    root = load_dataset_paths().root(DatasetName.FOURTU)
    with open(LINES_4TU_CSV, newline="", encoding="utf-8") as fh:
        lines = [r for r in csv.DictReader(fh) if r["source"] == "SLX"]
    for r in [lines[i] for i in sorted(rng.choice(len(lines), 5, replace=False))]:
        act, p = r["line"].split("/")
        img = make_detection_image(load_fourtu_radargram(root, act, p), CFG)
        tiles.append(img.tile_image(0))
        titles.append("4TU r0 " + r["line"])
    tile_grid(tiles, titles, boxes, path, ncol=5)


def main() -> None:
    below_normal()
    FIG.mkdir(parents=True, exist_ok=True)
    all_rows, imgs = [], {}
    for m in MODELS:
        for met, img in synthetic_lines(m):
            all_rows.append(met)
            imgs[(m, met["condition"])] = (img, met)
        print(m, "done", flush=True)

    boxes = d0_boxes()
    hits_y0 = {b["model"] for b in boxes if b["condition"] == "Y0"}

    def boxes_for(cond: str) -> dict[int, list]:
        out: dict[int, list] = {}
        colors = {"pipe_hyperbola": "#1baf7a", "manhole": "#e34948", "cavity": "#eda100",
                  "pipe_band": "#6250d6"}
        for b in boxes:
            if b["condition"] != cond:
                continue
            i = MODELS.index(b["model"])
            out.setdefault(i, []).append((float(b["x1"]), float(b["y1"]), float(b["x2"]),
                                          float(b["y2"]), f"f{b['fold']} {b['class'][:4]} {float(b['score']):.2f}",
                                          colors.get(b["class"], "#e34948")))
        return out

    for cond, name in (("Y0", "y0_tiles.png"), ("Y1_0.5P", "y1_05p_tiles.png"),
                       ("Y1_2P", "y1_2p_tiles.png")):
        tiles = [imgs[(m, cond)][0].tile_image(0) for m in MODELS]
        apex = [(imgs[(m, cond)][1]["apex_row_px"], imgs[(m, cond)][1]["apex_col_px"]) for m in MODELS]
        titles = [f"{m} {cond} H{imgs[(m, cond)][1]['content_h_px']}" for m in MODELS]
        tile_grid(tiles, titles, boxes_for(cond), FIG / name, ncol=5, apex=apex)
    size_scatter([r for r in all_rows if r["condition"] == "Y0"], hits_y0, FIG / "size_scatter.png")
    reference_tiles(FIG / "reference_tiles.png")

    # per-condition box counts from the D0 record
    count: dict[tuple[str, str], dict[str, int]] = {}
    for b in boxes:
        count.setdefault((b["condition"], b["class"]), {"n": 0})
        count[(b["condition"], b["class"])]["n"] += 1
        cat = b["category"] or "none"
        count[(b["condition"], b["class"])][cat] = count[(b["condition"], b["class"])].get(cat, 0) + 1

    tex = json.loads(TEXTURE_JSON.read_text(encoding="utf-8"))
    ref = {
        "aihub_g1.0_median": {k: v[0] for k, v in tex["aihub_reference"]["1.0"].items()
                              if isinstance(v, list)},
        "4tu_r0_median": tex["recipes"]["r0"]["4tu"]["median"],
    }
    summary = {
        "seed": SEED,
        "reference": ref,
        "d0_box_counts": {f"{c}|{k}": v for (c, k), v in sorted(count.items())},
        "y0_hit_models": sorted(hits_y0),
        "rows": all_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    keys = ["model", "condition", "content_w_px", "content_h_px", "pad_right_px", "pad_bottom_px",
            "content_area_frac", "period_l1_px", "compression", "zero_px", "apex_row_px",
            "pipe_frac_of_content", "floor_pre_db", "floor_post_db", "agc_floor_lift_db",
            "apex_post_over_clip", "visible_off_pipe_frac", "visible_pipe_frac",
            "top_band_mean_dev", "apex_gray_dev"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["w6_P", "h6_P", "ap6", "w20_P", "h20_P", "ap20", "bot20",
                           "R_h", "R_h_top15", "rho_bt", "std", "sat_frac"])
        for r in all_rows:
            b6, b20 = r["box"]["-6"], r["box"]["-20"]
            w.writerow([r[k] if not isinstance(r[k], float) else round(r[k], 4) for k in keys] + [
                round(b6["w_P"], 2) if b6 else "", round(b6["h_P"], 2) if b6 else "",
                int(b6["aperture_limited"]) if b6 else "",
                round(b20["w_P"], 2) if b20 else "", round(b20["h_P"], 2) if b20 else "",
                int(b20["aperture_limited"]) if b20 else "", int(b20["bottom_limited"]) if b20 else "",
                *(round(r["stats"][k], 4) if r["stats"][k] is not None else ""
                  for k in ("R_h", "R_h_top15", "rho_bt", "std", "sat_frac"))])
    print(json.dumps({"status": "ok", "rows": len(all_rows), "json": str(OUT_JSON)}))


if __name__ == "__main__":
    main()
