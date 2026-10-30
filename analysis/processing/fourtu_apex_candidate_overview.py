"""Overview figure of 4TU candidates on display images (AGC, detection-branch style).

Reads docs/processing/fourtu_apex_candidates_v1.csv; shows the given lines with auto
candidates (circles) and manual_v0 candidates (crosses). Display only: AGC is never used
for picks. Run:
  conda run -n gpr-dev python -I analysis/processing/fourtu_apex_candidate_overview.py --lines 01.1/Path1,010.14/Path2 --out name.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.preprocess.filters import agc  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
OUT = Path(__file__).resolve().parents[2] / "docs" / "processing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidates = pd.read_csv(
        OUT / "fourtu_apex_candidates_v1.csv", dtype={"activity_id": str}
    )
    keys = [tuple(s.split("/", 1)) for s in args.lines.split(",")]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.2 * len(keys), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (act, path) in zip(axes, keys, strict=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg)
        img = agc(inputs.branch.background_removed, rg.dt_ns, rg.f_c_hz)
        lim = np.percentile(np.abs(img), 99)
        ax.imshow(
            img,
            aspect="auto",
            cmap="gray",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
        )
        s = candidates[
            (candidates["activity_id"] == act) & (candidates["path"] == path)
        ]
        auto, manual = s[s["source"] == "auto_v1"], s[s["source"] == "manual_v0"]
        ax.scatter(
            auto["trace"],
            auto["sample"],
            s=60,
            facecolors="none",
            edgecolors="#e8590c",
            lw=1.4,
            label=f"auto_v1 ({len(auto)})",
        )
        ax.scatter(
            manual["trace"],
            manual["sample"],
            s=50,
            marker="x",
            color="#1c7ed6",
            lw=1.6,
            label=f"manual_v0 ({len(manual)})",
        )
        if inputs.zero.t_zero_ns is not None:
            ax.axhline(
                inputs.zero.t_zero_ns / rg.dt_ns, color="#2f9e44", lw=0.8, ls="--"
            )
        ax.set_title(
            f"{act} {path} (eps {rg.eps_r_meta:g}, dt {rg.dt_ns:g})", fontsize=9
        )
        ax.set_xlabel("trace")
        ax.set_ylabel("sample")
        ax.legend(fontsize=7, loc="lower right")
    fig.savefig(OUT / args.out, dpi=110)


if __name__ == "__main__":
    main()
