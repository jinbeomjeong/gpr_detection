"""M2-9 C: intra-rater re-read of the apex-candidate review (30 of 169), blinded.

Step 1 only (sampling + re-rendering). This is the ONLY script that reads the first
review sheet before the re-read sheet is locked. It prints no mapping and no sample
composition, so the reader sees nothing but the new ids.

Sampling (stratified by first-read confidence x type):
  * pool = 169 first-read candidates minus G001 (its first-read row was shown on
    screen while inspecting the csv header on 2026-11-07, so it is excluded).
  * 10 per confidence level (high / medium / low).
  * inside a level: 1 per type present (if the level has <= 10 types), the rest by
    largest-remainder allocation proportional to the remaining pool per type,
    capped by availability; members drawn at random.
  * new order: random permutation -> R01..R30.
Seed: 20261107 (numpy default_rng).

Rendering: the same patch/draw code and the same depth-path preprocessing as
m29c_gallery.py (seed 20261106 sample), only the panel title changes to the new id.
A pixel check re-renders every sampled panel with its original title and compares
it with patches/<G>.png; only the match count is printed.

Outputs (git-ignored, data/processing/m29c_gallery/):
  reread_key.csv            R id -> G id, first type/confidence (do not open before lock)
  reread_sheet.csv          R id only, empty type/confidence/comment
  patches_reread/R??.png    one panel per re-read candidate

Run: conda run -n gpr-dev python -I analysis/processing/m29c_reread_sample.py
"""

import importlib.util
import io
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.preprocess.filters import agc  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
GAL = PROJECT / "data" / "processing" / "m29c_gallery"
SEED = 20261107
PER_LEVEL = 10
LEVELS = ["high", "medium", "low"]
EXCLUDE = {"G001"}


def load_gallery_module():
    spec = importlib.util.spec_from_file_location(
        "m29c_gallery", Path(__file__).resolve().parent / "m29c_gallery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    types = sorted(counts)
    alloc = {t: 0 for t in types}
    if len(types) <= n:
        for t in types:
            alloc[t] = 1
    left = n - sum(alloc.values())
    rem = {t: counts[t] - alloc[t] for t in types}
    total = sum(rem.values())
    if left > 0 and total > 0:
        quota = {t: left * rem[t] / total for t in types}
        add = {t: min(int(math.floor(quota[t])), rem[t]) for t in types}
        order = sorted(types, key=lambda t: (-(quota[t] - math.floor(quota[t])), t))
        k = left - sum(add.values())
        for t in order:
            if k <= 0:
                break
            if add[t] < rem[t]:
                add[t] += 1
                k -= 1
        for t in types:
            alloc[t] += add[t]
    return alloc


def render(gal_mod, data, t_c, t_zero, label) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    gal_mod.draw(ax, data, t_c, t_zero, label)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf


def main() -> None:
    gal = load_gallery_module()
    first = pd.read_csv(GAL / "review_sheet_locked.csv")
    first = first[~first["blind_id"].isin(EXCLUDE)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    picked = []
    for level in LEVELS:
        sub = first[first["confidence"] == level]
        alloc = allocate(sub["type"].value_counts().to_dict(), PER_LEVEL)
        for t, k in sorted(alloc.items()):
            if k == 0:
                continue
            ids = sub.loc[sub["type"] == t, "blind_id"].to_numpy()
            picked.extend(rng.choice(ids, size=k, replace=False).tolist())

    order = rng.permutation(len(picked))
    picked = [picked[i] for i in order]
    key = pd.DataFrame(
        {"reread_id": [f"R{i + 1:02d}" for i in range(len(picked))], "blind_id": picked}
    )
    key = key.merge(
        first.rename(columns={"type": "first_type", "confidence": "first_confidence"})[
            ["blind_id", "first_type", "first_confidence"]
        ],
        on="blind_id",
        validate="one_to_one",
    )
    key["seed"] = SEED
    key["method"] = "conf x type stratified, 10/level, 1 per type then largest remainder"

    sample = pd.read_csv(GAL / "sample_key.csv", dtype={"activity_id": str})
    sample = sample[sample["blind_id"].isin(picked)]
    (GAL / "patches_reread").mkdir(exist_ok=True)
    r_of = dict(zip(key["blind_id"], key["reread_id"]))
    n_match = 0
    for (act, path), group in sample.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(gal.BASE, act, path)
        inputs = prepare_depth_inputs(rg, DepthConfig())
        img = agc(inputs.branch.background_removed, rg.dt_ns, rg.f_c_hz)
        t_zero = inputs.zero.t_zero_ns
        for row in group.itertuples():
            t_c = float(row.sample) * rg.dt_ns
            data = gal.patch(rg, img, row.trace, t_c)
            check = mpimg.imread(render(gal, data, t_c, t_zero, row.blind_id))
            orig = mpimg.imread(GAL / "patches" / f"{row.blind_id}.png")
            if check.shape == orig.shape and np.array_equal(check, orig):
                n_match += 1
            buf = render(gal, data, t_c, t_zero, r_of[row.blind_id])
            (GAL / "patches_reread" / f"{r_of[row.blind_id]}.png").write_bytes(
                buf.getvalue()
            )

    key.to_csv(GAL / "reread_key.csv", index=False)
    pd.DataFrame(
        {"reread_id": key["reread_id"], "type": "", "confidence": "", "comment": ""}
    ).sort_values("reread_id").to_csv(GAL / "reread_sheet.csv", index=False)
    print(f"n={len(key)} seed={SEED} pixel_identical_to_first_render={n_match}/{len(key)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
