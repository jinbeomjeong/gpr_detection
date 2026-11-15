"""M3-3 D0 follow-up (2): Y0 visible width after AGC and the Y1 column artefact.

Diagnostic only, CPU, single process, BelowNormal, no detector.

For every gprMax model M01-M20:
  * Y0: -6 / -20 dB box of the *detection* amplitudes (after r0 AGC) seeded
    at the apex, in L1 periods (what the detector sees), next to the
    pre-AGC pipe-only box of m33_y0_detect_diag.
  * Y1 cuts (0.5 / 1 / 2 / 3 P, period held from Y0 as in the D0 runner):
    column-structure share R_v of the grey content for
      (a) D0 path: cut record -> r0 image (period override),
      (b) same with gain 'none' (no AGC),
      (c) full record -> r0 amplitudes -> cut -> grey (process, then cut),
    plus the AGC window / record length ratio and the D0 box count per line.
R_v = n_rows * sum_c (colmean_c - mean)^2 / sum (q - mean)^2 (column analogue
of the stage-A R_h); 1/n_rows for white noise, 1 for pure vertical stripes.
Outputs: analysis/processing/m33_y1_artifact_diag.{json,csv},
docs/processing/figures/m33_y0_diag/y1_paths_M04.png.
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.io.synthetic import load_synthetic, load_synthetic_truth  # noqa: E402
from gpr_detection.preprocess.filters import AGC_WINDOW_PERIODS  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DEFAULT_DETECTION_IMAGE_CONFIG,
    _line_amplitudes,
    make_detection_image,
    resample_area,
    to_gray,
)

SYN = ROOT / "data" / "synthetic_gprmax"
D0 = ROOT / "runs" / "eval" / "m33_d0"
OUT_JSON = ROOT / "analysis" / "processing" / "m33_y1_artifact_diag.json"
OUT_CSV = ROOT / "analysis" / "processing" / "m33_y1_artifact_diag.csv"
FIG = ROOT / "docs" / "processing" / "figures" / "m33_y0_diag"
MODELS = tuple(f"M{i:02d}" for i in range(1, 21))
CUTS = (0.5, 1.0, 2.0, 3.0)
CFG = DEFAULT_DETECTION_IMAGE_CONFIG
CFG_NOGAIN = replace(CFG, gain_mode="none", recipe_id="diag_nogain")


def below_normal() -> None:
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def r_v(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    a = q - q.mean()
    total = float(np.sum(a**2))
    return q.shape[0] * float(np.sum(a.mean(axis=0) ** 2)) / total if total > 0 else float("nan")


def grad_ratio(q: np.ndarray) -> float:
    """Horizontal over vertical grey-gradient energy (vertical stripes raise it)."""
    q = np.asarray(q, dtype=np.float64)
    gx = float(np.mean(np.diff(q, axis=1) ** 2))
    gy = float(np.mean(np.diff(q, axis=0) ** 2))
    return gx / gy if gy > 0 else float("nan")


def low_adjacent_corr(q: np.ndarray, rows: int = 25, threshold: float = 0.5) -> float:
    """Share of adjacent column pairs (top ``rows`` rows) with correlation < threshold.

    Column blocks with independent contrast (Y1 artefact) give low pairs;
    a smooth arc keeps neighbouring columns correlated. Flat pairs are skipped.
    """
    q = np.asarray(q, dtype=np.float64)[:rows]
    a = q - q.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(a**2, axis=0))
    ok = (norm[:-1] > 1e-6) & (norm[1:] > 1e-6)
    if not ok.any():
        return float("nan")
    corr = np.sum(a[:, :-1] * a[:, 1:], axis=0)[ok] / (norm[:-1] * norm[1:])[ok]
    return float(np.mean(corr < threshold))


def r_h(q: np.ndarray) -> float:
    return r_v(np.asarray(q).T)


def seed_box_periods(amp: np.ndarray, shape: tuple[int, int], apex: tuple[float, float],
                     spp: float, tpp: float, p_l1: float) -> dict:
    env = resample_area(np.abs(hilbert(np.asarray(amp, dtype=np.float64), axis=0)), shape)
    H, W = shape
    ar, ac = min(H - 1, int(apex[0] / spp)), min(W - 1, int(apex[1] / tpp))
    r0, r1 = max(0, ar - int(p_l1)), min(H, ar + int(p_l1) + 1)
    c0, c1 = max(0, ac - 3), min(W, ac + 4)
    sub = env[r0:r1, c0:c1]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    seed = (r0 + k[0], c0 + k[1])
    out = {}
    for level in (-6, -20):
        lab, _ = ndimage.label(env >= env[seed] * 10 ** (level / 20))
        rows, cols = np.nonzero(lab == lab[seed])
        out[str(level)] = {
            "w_P": (cols.max() - cols.min() + 1) / p_l1,
            "h_P": (rows.max() - rows.min() + 1) / p_l1,
            "aperture_limited": bool(cols.min() == 0 or cols.max() == W - 1),
            "frac_of_content": float(np.mean(lab == lab[seed])),
        }
    return out


def gray_content(img) -> np.ndarray:
    return img.line_image[: img.content_height_px].astype(np.float64)


def box_counts() -> dict[tuple[str, str], int]:
    n: dict[tuple[str, str], int] = {}
    for k in range(5):
        with open(D0 / f"fold{k}" / "synthetic_boxes.csv", newline="", encoding="utf-8") as fh:
            for b in csv.DictReader(fh):
                n[(b["model"], b["condition"])] = n.get((b["model"], b["condition"]), 0) + 1
    return n


def main() -> None:
    below_normal()
    counts = box_counts()
    rows, y0_rows = [], []
    fig_payload = None
    for m in MODELS:
        rg = load_synthetic(m, root=SYN)
        truth = load_synthetic_truth(m, root=SYN)
        apex_sample = (truth.apex_envelope_peak_ns - float(rg.metadata.get("time_of_sample0_ns", 0.0))) / rg.dt_ns
        apex = (apex_sample, float(truth.apex_trace_index))
        img0 = make_detection_image(rg, CFG)
        amps0 = _line_amplitudes(rg, CFG).amplitudes
        H, W = img0.content_height_px, img0.width_px
        spp, tpp = img0.grid_map.samples_per_px, img0.grid_map.traces_per_px
        p_l1 = img0.period_samples / spp
        post = seed_box_periods(amps0, (H, W), apex, spp, tpp, p_l1)
        g0 = gray_content(img0)
        y0_rows.append({"model": m, "post_agc_box": post, "line_width_P": W / p_l1,
                        "R_v": r_v(g0), "R_h": r_h(g0), "G": grad_ratio(g0), "L": low_adjacent_corr(g0),
                        "agc_window_over_record": (AGC_WINDOW_PERIODS * 1e9 / rg.f_c_hz / rg.dt_ns) / rg.n_samples})
        period = img0.period_samples
        window_samples = AGC_WINDOW_PERIODS * 1e9 / rg.f_c_hz / rg.dt_ns
        for f in CUTS:
            cond = f"Y1_{f:g}P"
            n_keep = min(rg.n_samples, int(np.floor(apex_sample + f * period)) + 1)
            cut = replace(rg, data=np.ascontiguousarray(rg.data[:n_keep]))
            a_img = make_detection_image(cut, CFG, period_samples_override=period)
            b_img = make_detection_image(cut, CFG_NOGAIN, period_samples_override=period)
            # (c) process the full record, then cut the amplitudes and map greys
            amps_c = amps0[:n_keep]
            shape = (a_img.content_height_px, a_img.width_px)
            clip = float(np.percentile(np.abs(amps_c), CFG.clip_percentile))
            c_gray = to_gray(resample_area(amps_c, shape), clip).astype(np.float64)
            ga, gb = gray_content(a_img), gray_content(b_img)
            rows.append({
                "model": m, "condition": cond, "n_keep": n_keep,
                "agc_window_over_record": window_samples / n_keep,
                "R_v_d0": r_v(ga), "R_v_nogain": r_v(gb), "R_v_full_then_cut": r_v(c_gray),
                "G_d0": grad_ratio(ga), "G_nogain": grad_ratio(gb),
                "G_full_then_cut": grad_ratio(c_gray),
                "L_d0": low_adjacent_corr(ga), "L_nogain": low_adjacent_corr(gb),
                "L_full_then_cut": low_adjacent_corr(c_gray),
                "R_h_d0": r_h(ga),
                "d0_boxes_5folds": counts.get((m, cond), 0),
            })
            if m == "M04" and f == 0.5:
                fig_payload = (a_img.tile_image(0), b_img.tile_image(0), c_gray, img0.tile_image(0))
        print(m, flush=True)

    # summary
    def med(key, cond=None):
        vals = [r[key] for r in rows if cond is None or r["condition"] == cond]
        return float(np.median(vals))

    by_cond = {}
    for f in CUTS:
        c = f"Y1_{f:g}P"
        by_cond[c] = {k: med(k, c) for k in ("R_v_d0", "R_v_nogain", "R_v_full_then_cut",
                                               "G_d0", "G_nogain", "G_full_then_cut",
                                               "L_d0", "L_nogain", "L_full_then_cut",
                                               "agc_window_over_record")}
        by_cond[c]["boxes"] = int(sum(r["d0_boxes_5folds"] for r in rows if r["condition"] == c))
    from PIL import Image

    aihub_g: dict[str, list[float]] = {}
    with open(ROOT / "analysis" / "processing" / "m32_stageA_aihub_ref.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["g1.0"]:
                q = np.asarray(Image.open(ROOT / "data" / "aihub_yolo_v1" / "images" / r["g1.0"]).convert("L"))
                aihub_g.setdefault(r["class"], []).append(grad_ratio(q))
                aihub_g.setdefault(r["class"] + "|L", []).append(low_adjacent_corr(q))
    aihub_g_med = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                       "n": len(v)} for k, v in aihub_g.items()}
    rho = spearmanr([r["L_d0"] for r in rows], [r["d0_boxes_5folds"] for r in rows],
                    nan_policy="omit")
    summary = {
        "y0": y0_rows,
        "y0_R_v_median": float(np.median([r["R_v"] for r in y0_rows])),
        "y0_G_median": float(np.median([r["G"] for r in y0_rows])),
        "y0_L_median": float(np.nanmedian([r["L"] for r in y0_rows])),
        "y1_by_condition": by_cond,
        "aihub_g1.0_G_by_class": aihub_g_med,
        "spearman_G_vs_boxes": {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(rows)},
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c, y0 = fig_payload
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    titles = ["Y0 (full record, r0)", "Y1 0.5P D0 path (cut -> r0)",
              "Y1 0.5P cut -> no AGC", "Y1 0.5P full r0 -> cut"]
    for ax, im, t in zip(axes, [y0, a, b, c], titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(t, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("M04: where the Y1 column texture comes from", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "y1_paths_M04.png", dpi=120)
    plt.close(fig)
    print(json.dumps({"status": "ok", "y1": by_cond, "spearman": summary["spearman_G_vs_boxes"],
                      "y0_G_median": summary["y0_G_median"], "y0_L_median": summary["y0_L_median"],
                      "aihub": aihub_g_med}))


if __name__ == "__main__":
    main()
