"""M1-7 pipe_band background false positives: figures 1 (per-class FP bars) and 2 (confusion heatmaps).

Reads runs/detect/<run>/eval/last_crop/confusion_matrix_conf0.25.csv (rows = predicted, cols = true,
background included; written by scripts/eval_detector.py --confusion-conf 0.25) and metrics.json
(instance counts, eval image count). Cross-checks the background->class counts against
analysis/detection/m17_seed_aggregate.json. CPU only.

Usage: python -I analysis/detection/m17_pipe_band_fp_plots.py [--root C:/workspace/gpr_detection]
Writes docs/experiments/figures/m17_pipe_band_fp_bars.png and m17_pipe_band_fp_confusion.png;
the last stdout line is a one-line JSON summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

# (label, strength, seed, run folder) in display order
RUNS = [
    ("강도 0\nseed 0", "0", 0, "m17_aug0"),
    ("강도 0\nseed 1", "0", 1, "m17_aug0_s1"),
    ("강도 0.5\nseed 0", "0.5", 0, "m17_aug0.5"),
    ("강도 1\nseed 0", "1", 0, "m17_aug1"),
    ("강도 1\nseed 1", "1", 1, "m17_aug1_s1"),
    ("강도 2\nseed 0", "2", 0, "m17_aug2"),
]
CLASSES = ["pipe_hyperbola", "pipe_band", "cavity", "manhole"]
# dataviz reference palette, categorical slots 1-4 (validated, light mode)
COLORS = {"pipe_hyperbola": "#2a78d6", "pipe_band": "#eb6834", "cavity": "#1baf7a", "manhole": "#eda100"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def read_cm(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "predicted\\true", path
    cols = rows[0][1:]
    pred = [r[0] for r in rows[1:]]
    mat = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=int)
    return pred, cols, mat


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    root = Path(a.root)
    fig_dir = root / "docs" / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        agg = json.loads((root / "analysis" / "detection" / "m17_seed_aggregate.json").read_text(encoding="utf-8"))
        data = []
        for label, s, seed, run in RUNS:
            d = root / "runs" / "detect" / run / "eval" / "last_crop"
            pred, cols, mat = read_cm(d / "confusion_matrix_conf0.25.csv")
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            inst = {c["name"]: c["instances"] for c in m["per_class"]}
            bg_col = cols.index("background")
            fp = {c: int(mat[pred.index(c), bg_col]) for c in CLASSES}
            ref = agg["runs"]["last_crop"][f"{s}|{seed}"]["confusion_conf0.25"]["bg_fp"]
            assert fp == {c: ref[c] for c in CLASSES}, (run, fp, ref)
            data.append({"label": label, "run": run, "strength": s, "seed": seed, "pred": pred, "cols": cols,
                         "mat": mat, "fp": fp, "inst": inst, "n_images": m["eval_images"]["count"]})
    except (OSError, KeyError, ValueError, AssertionError) as e:
        print(json.dumps({"status": "input_error", "error": repr(e)}))
        return 2

    n_img = {d["n_images"] for d in data}
    insts = {json.dumps(d["inst"], sort_keys=True) for d in data}
    assert len(n_img) == 1 and len(insts) == 1
    n_img = n_img.pop()
    inst = data[0]["inst"]
    ratios = []
    for d in data:
        other = max(d["fp"][c] for c in CLASSES if c != "pipe_band")
        d["ratio_vs_max_other"] = d["fp"]["pipe_band"] / other
        ratios.append(d["ratio_vs_max_other"])

    # ---------------- figure 1: grouped bars + FP/GT ratio ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.6), height_ratios=[1.35, 1], sharex=True,
                                   facecolor=SURFACE)
    x = np.arange(len(data))
    w = 0.19
    for k, c in enumerate(CLASSES):
        off = (k - 1.5) * w
        vals = [d["fp"][c] for d in data]
        bars = ax1.bar(x + off, vals, w * 0.9, color=COLORS[c], label=f"{c} (정답 n={inst[c]})", zorder=3)
        # GT instance count of the class as a short reference tick over each bar
        for xi in x:
            ax1.hlines(inst[c], xi + off - w * 0.45, xi + off + w * 0.45, color=INK, linewidth=1.6, zorder=4)
        if c == "pipe_band":
            for b, v in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width() / 2, v + 12, f"{v}", ha="center", va="bottom", fontsize=8.5,
                         color=INK, zorder=5)
        ratios_c = [d["fp"][c] / inst[c] for d in data]
        ax2.bar(x + off, ratios_c, w * 0.9, color=COLORS[c], zorder=3)
        if c == "pipe_band":
            for xi, r in zip(x + off, ratios_c):
                ax2.text(xi, r + 0.03, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, zorder=5)
    ax1.plot([], [], color=INK, linewidth=1.6, label="정답 인스턴스 수(막대 위 가로선, 클래스별)")
    for xi, d in zip(x, data):
        ax1.text(xi, 1000, f"×{d['ratio_vs_max_other']:.1f}", ha="center", va="bottom", fontsize=10,
                 color=INK, fontweight="bold")
    ax1.text(len(data) - 0.5, 1075, "×값 = pipe_band 오검출 ÷ 같은 실행의 다른 3클래스 중 최대 오검출",
             ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax1.set_ylim(0, 1420)
    ax1.set_yticks(range(0, 1001, 200))
    ax1.set_ylabel("배경 → 클래스 오검출 수 (개)", color=INK, fontsize=10)
    ax1.set_title(f"M1-7 클래스별 배경 오검출 (conf 0.25, 매칭 IoU 0.45) — fold 0 검증 crop {n_img:,}장, "
                  f"last.pt, YOLO11m imgsz 256\npipe_band 오검출은 같은 실행의 다른 클래스 최대값의 "
                  f"×{min(ratios):.1f}–×{max(ratios):.1f}", fontsize=11, color=INK, loc="left")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    ax2.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax2.text(len(data) - 0.5, 1.0, "오검출 = 정답 수", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("오검출 수 / 정답 인스턴스 수", color=INK, fontsize=10)
    ax2.set_xticks(x, [d["label"] for d in data])
    ax2.set_xlabel("증강 강도 · 학습 seed (각 1회 학습, fold 0)", color=INK, fontsize=10)
    for ax in (ax1, ax2):
        style_axes(ax)
    fig.tight_layout()
    out1 = fig_dir / "m17_pipe_band_fp_bars.png"
    fig.savefig(out1, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figure 2: confusion heatmaps (strength 0, seed 0 and 1) ----------------
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    reps = [d for d in data if d["strength"] == "0"]
    vmax = max(int(d["mat"].max()) for d in reps)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=SURFACE, gridspec_kw={"wspace": 0.42})
    for ax, d in zip(axes, reps):
        mat = d["mat"]
        im = ax.imshow(mat, cmap=cmap, norm=norm)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "#ffffff" if norm(v) > 0.55 else INK
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=9.5, color=color)
        bi, bj = d["pred"].index("pipe_band"), d["cols"].index("background")
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=COLORS["pipe_band"],
                                   linewidth=2.5))
        ax.set_xticks(range(len(d["cols"])), d["cols"], rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(d["pred"])), d["pred"], fontsize=9)
        ax.set_xlabel("정답 클래스 (열)", color=INK, fontsize=10)
        ax.set_ylabel("예측 클래스 (행)", color=INK, fontsize=10)
        ax.set_title(f"강도 0 · seed {d['seed']} ({d['run']})", fontsize=10.5, color=INK)
        ax.tick_params(colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("개수 (색은 제곱근 눈금, 칸 숫자는 원값)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle(f"conf 0.25 혼동 행렬(정규화 없음, background 포함, 매칭 IoU 0.45) — fold 0 검증 crop {n_img:,}장, "
                 f"last.pt\n정답 인스턴스: " + ", ".join(f"{c} {inst[c]}" for c in CLASSES)
                 + " · 주황 테두리 = 배경→pipe_band 오검출", fontsize=10.5, color=INK, x=0.02, ha="left")
    out2 = fig_dir / "m17_pipe_band_fp_confusion.png"
    fig.savefig(out2, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    summary = {"status": "ok", "n_images": n_img, "instances": inst,
               "ratio_band_vs_max_other": {d["run"]: round(d["ratio_vs_max_other"], 3) for d in data},
               "ratio_min": round(min(ratios), 3),
               "fp_over_gt_band": {d["run"]: round(d["fp"]["pipe_band"] / inst["pipe_band"], 3) for d in data},
               "out": [str(out1), str(out2)]}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
