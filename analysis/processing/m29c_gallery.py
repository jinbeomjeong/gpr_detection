"""M2-9 C step 1: blinded apex-candidate gallery for manual review (no ground truth).

Sample (user decision 2026-11-06): all manual_v0 candidates (39) + 10 auto_v1
candidates per site, stratified by site x candidate-depth tertile. Candidate depth
d_c = v_meta * tau_ns / 2 with v_meta = c / sqrt(eps_r_used) (metadata, not truth).
Tertile edges are computed per site over that site's auto candidates; each tertile
gets 3 candidates and one extra goes to a seeded random tertile. Ground-truth depth
tables and M2-8 matching results are NOT read.

Display: depth-path preprocessing (prepare_depth_inputs, DepthConfig defaults:
DC, Gaussian dewow, band-pass, median background) then the detection-branch AGC for
display only. Window: candidate x +-1.5 m, t_c - 10 ns .. t_c + 20 ns (NaN padding
outside the record so every panel has the same physical scale). No picks, no
fitted or metadata hyperbola is drawn (curvature must not steer the review).
Panels are labelled by a shuffled blind id only.

Outputs (git-ignored): data/processing/m29c_gallery/
  sample_key.csv          blind id -> candidate (do not open before classifying)
  review_sheet.csv        blind id only, empty type/confidence columns
  patches/<id>.png        one panel per candidate
  sheets/sheet_XX.png     3 x 3 montages for the review

Run (single process, BLAS threads 1):
  conda run -n gpr-dev python -I analysis/processing/m29c_gallery.py
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.filters import agc  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
CANDIDATES = PROJECT / "docs" / "processing" / "fourtu_apex_candidates_v1.csv"
LINES = PROJECT / "docs" / "processing" / "fourtu_lines_v1.csv"
OUT = PROJECT / "data" / "processing" / "m29c_gallery"

SEED = 20261106
PER_SITE = 10
N_BINS = 3
HALF_WIDTH_M = 1.5
T_BEFORE_NS = 10.0
T_AFTER_NS = 20.0
SHEET_ROWS, SHEET_COLS = 3, 3


def candidate_depth(frame: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    eps = frame.merge(
        lines[["activity_id", "path", "eps_r_used"]],
        on=["activity_id", "path"],
        how="left",
        validate="many_to_one",
    )["eps_r_used"]
    v_meta = eps.map(velocity_from_eps).to_numpy()
    return pd.Series(v_meta * frame["tau_ns"].to_numpy() / 2.0, index=frame.index)


def stratified_sample(cands: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    auto = cands[(cands["source"] == "auto_v1") & cands["d_cand_m"].notna()]
    edges = {}
    picked = []
    for site, group in auto.groupby("site", sort=True):
        q1, q2 = np.quantile(group["d_cand_m"], [1 / 3, 2 / 3])
        edges[site] = (float(q1), float(q2))
        bins = np.digitize(group["d_cand_m"], [q1, q2])  # 0, 1, 2
        counts = [PER_SITE // N_BINS] * N_BINS
        counts[int(rng.integers(N_BINS))] += PER_SITE - sum(counts)
        for b in range(N_BINS):
            pool = group[bins == b]
            take = min(counts[b], len(pool))
            idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            picked.extend(int(i) for i in idx)
    sample = cands.loc[picked].copy()
    manual = cands[cands["source"] == "manual_v0"].copy()
    out = pd.concat([manual, sample])

    def bin_of(row: pd.Series) -> float:
        if pd.isna(row["d_cand_m"]) or row["site"] not in edges:
            return math.nan
        return float(np.digitize(row["d_cand_m"], edges[row["site"]]))

    out["depth_bin"] = out.apply(bin_of, axis=1)
    out["bin_q1_m"] = out["site"].map(lambda s: edges.get(s, (math.nan,) * 2)[0])
    out["bin_q2_m"] = out["site"].map(lambda s: edges.get(s, (math.nan,) * 2)[1])
    return out


def patch(rg, img: np.ndarray, trace: float, t_c: float):
    dt, dx = rg.dt_ns, rg.dx_m
    half = int(round(HALF_WIDTH_M / dx))
    n_before = int(round(T_BEFORE_NS / dt))
    n_after = int(round(T_AFTER_NS / dt))
    k_c = int(round(t_c / dt))
    i_c = int(round(trace))
    out = np.full((n_before + n_after + 1, 2 * half + 1), np.nan)
    k_lo, k_hi = k_c - n_before, k_c + n_after
    i_lo, i_hi = i_c - half, i_c + half
    sk0, sk1 = max(k_lo, 0), min(k_hi, img.shape[0] - 1)
    si0, si1 = max(i_lo, 0), min(i_hi, img.shape[1] - 1)
    out[sk0 - k_lo : sk1 - k_lo + 1, si0 - i_lo : si1 - i_lo + 1] = img[
        sk0 : sk1 + 1, si0 : si1 + 1
    ]
    return out


def draw(ax, data: np.ndarray, t_c: float, t_zero: float | None, label: str) -> None:
    finite = np.abs(data[np.isfinite(data)])
    lim = float(np.percentile(finite, 99)) if finite.size else 1.0
    ax.imshow(
        np.ma.masked_invalid(data),
        cmap="gray",
        vmin=-lim,
        vmax=lim,
        aspect="auto",
        interpolation="nearest",
        extent=(-HALF_WIDTH_M, HALF_WIDTH_M, t_c + T_AFTER_NS, t_c - T_BEFORE_NS),
    )
    ax.set_facecolor("#c8d6e5")
    kw = dict(color="#e03131", ms=7, clip_on=False, zorder=5)
    ax.plot([-HALF_WIDTH_M], [t_c], marker=">", **kw)
    ax.plot([HALF_WIDTH_M], [t_c], marker="<", **kw)
    ax.plot([0.0], [t_c - 3.0], marker="v", **kw)
    if t_zero is not None and t_zero > t_c - T_BEFORE_NS:
        ax.axhline(t_zero, color="#2f9e44", lw=0.8, ls="--")
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x - x_c (m)", fontsize=7)
    ax.set_ylabel("t (ns)", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    (out / "patches").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    cands = pd.read_csv(CANDIDATES, dtype={"activity_id": str})
    lines = pd.read_csv(LINES, dtype={"activity_id": str})
    # Blind: the straight-ray QC velocity of the selection step is a curvature value.
    cands = cands.drop(columns=["v_straight_qc"])
    cands["cand_index"] = cands.index
    cands["d_cand_m"] = candidate_depth(cands, lines)

    rng = np.random.default_rng(SEED)
    sample = stratified_sample(cands, rng)
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "blind_id", [f"G{i + 1:03d}" for i in range(len(sample))])

    rows = []
    patches = {}
    for (act, path), group in sample.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg, DepthConfig())
        img = agc(inputs.branch.background_removed, rg.dt_ns, rg.f_c_hz)
        t_zero = inputs.zero.t_zero_ns
        for row in group.itertuples():
            t_c = float(row.sample) * rg.dt_ns
            patches[row.blind_id] = (patch(rg, img, row.trace, t_c), t_c, t_zero)
            rows.append(
                {
                    "blind_id": row.blind_id,
                    "dt_ns": rg.dt_ns,
                    "dx_m": rg.dx_m,
                    "eps_r_used": rg.eps_r_meta,
                    "t_zero_detected_ns": t_zero,
                    "zero_rule": str(inputs.zero.zero_rule),
                }
            )
    sample = sample.merge(pd.DataFrame(rows), on="blind_id", validate="one_to_one")
    sample = sample.sort_values("blind_id").reset_index(drop=True)
    sample.attrs["seed"] = SEED
    sample.to_csv(out / "sample_key.csv", index=False)
    pd.DataFrame(
        {
            "blind_id": sample["blind_id"],
            "type": "",
            "confidence": "",
            "comment": "",
        }
    ).to_csv(out / "review_sheet.csv", index=False)

    ids = list(sample["blind_id"])
    for bid in ids:
        data, t_c, t_zero = patches[bid]
        fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
        draw(ax, data, t_c, t_zero, bid)
        fig.savefig(out / "patches" / f"{bid}.png", dpi=110)
        plt.close(fig)

    per_sheet = SHEET_ROWS * SHEET_COLS
    for s in range(math.ceil(len(ids) / per_sheet)):
        chunk = ids[s * per_sheet : (s + 1) * per_sheet]
        fig, axes = plt.subplots(
            SHEET_ROWS, SHEET_COLS, figsize=(13.5, 12.0), constrained_layout=True
        )
        for ax, bid in zip(axes.ravel(), chunk + [None] * (per_sheet - len(chunk))):
            if bid is None:
                ax.axis("off")
                continue
            data, t_c, t_zero = patches[bid]
            draw(ax, data, t_c, t_zero, bid)
        fig.savefig(out / "sheets" / f"sheet_{s + 1:02d}.png", dpi=85)
        plt.close(fig)

    counts = sample.groupby(["site", "source"]).size()
    print(counts.to_string())
    print(f"n={len(sample)} seed={SEED}")


if __name__ == "__main__":
    main()
