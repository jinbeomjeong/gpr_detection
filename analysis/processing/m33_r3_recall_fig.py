"""Figure for docs/processing/m33_r3_band_recall_review.md (read-only).

Reads analysis/processing/m33_r3_recall_diag_units.csv (written by
m33_r3_recall_diag.py) and draws, for the P4 scope (IDS main_400, A0,
main cell, L/W_eff < 0.5):
  (a) length-weighted pipe_band recall vs SNR for four placement groups
  (b) recall at 18 dB by the truth top row in the L1 image, with the AIHub
      lpipe label top-row range (p5/p50/p95 46/57/79 px, min 36.9 px).
No AIHub pixels are drawn.
Run: conda run -n gpr-dev python -I -B analysis/processing/m33_r3_recall_fig.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "analysis" / "processing" / "m33_r3_recall_diag_units.csv"
OUT = ROOT / "docs" / "processing" / "figures" / "m33_r3_recall" / "recall_by_placement.png"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # reference palette slots 1-4
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main() -> None:
    rows = [
        r
        for r in csv.DictReader(UNITS.open(encoding="utf-8"))
        if r["dataset"] == "gz_ids"
        and r["freq"] == "main_400"
        and r["model_group"] == "A0"
        and r["main_cell"] == "True"
        and r["lw"] in ("lt033", "033_05")
    ]

    def rec(us):
        return 100 * sum(int(r["cov_band50"]) for r in us) / max(1, sum(int(r["L"]) for r in us))

    def group(r):
        long_ = float(r["L_periods_px"]) >= 12.4
        deep = float(r["truth_y1_px"]) >= 37
        return {
            (True, True): "long & deep",
            (True, False): "long & shallow",
            (False, True): "short & deep",
            (False, False): "short & shallow",
        }[(long_, deep)]

    names = ("long & deep", "long & shallow", "short & deep", "short & shallow")
    by = defaultdict(list)
    for r in rows:
        by[(group(r), float(r["snr"]))].append(r)
    npos = {g: len({r["position_id"] for r in rows if group(r) == g}) for g in names}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    snrs = (0.0, 6.0, 12.0, 18.0)
    for color, g in zip(SERIES, names):
        ys = [rec(by[(g, s)]) for s in snrs]
        ax1.plot(snrs, ys, color=color, lw=2, marker="o", ms=5, label=f"{g} ({npos[g]} pos.)")
        ax1.annotate(g, (18, ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax1.set_xticks(snrs)
    ax1.set_xlim(-1, 25)
    ax1.set_xlabel("SNR (dB, injection definition)")
    ax1.set_ylabel("pipe_band recall (%, length-weighted)")
    ax1.set_title("(a) P4 scope: recall vs SNR by placement", color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax1.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    edges = (0, 37, 46, 57, 79, 120)
    labels, vals, ns = [], [], []
    r18 = [r for r in rows if float(r["snr"]) == 18.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        us = [r for r in r18 if lo <= float(r["truth_y1_px"]) < hi]
        labels.append(f"{lo}-{hi}" if hi < 120 else f">={lo}")
        vals.append(rec(us))
        ns.append(len({r["position_id"] for r in us}))
    xs = range(len(vals))
    ax2.axvspan(1.5, 3.5, color="#cde2fb", lw=0, zorder=0)
    ax2.text(2.5, max(vals) * 1.42, "AIHub lpipe top row p5-p95 (46-79 px)", ha="center", va="top", color=INK2, fontsize=8)
    ax2.bar(xs, vals, width=0.72, color=SERIES[0], edgecolor=SURFACE, lw=2, zorder=2)
    for x, v, n in zip(xs, vals, ns):
        ax2.text(x, v + 1, f"{v:.0f}%\n{n} pos.", ha="center", va="bottom", color=INK2, fontsize=8)
    ax2.set_xticks(list(xs), labels)
    ax2.set_ylim(0, max(vals) * 1.45)
    ax2.set_xlabel("truth box top row in the L1 image (px)")
    ax2.set_ylabel("recall at 18 dB (%)")
    ax2.set_title("(b) P4 scope, 18 dB: recall by box top row", color=INK, loc="left")
    ax2.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=1)
    ax2.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print({"status": "ok", "out": str(OUT)})


if __name__ == "__main__":
    main()
