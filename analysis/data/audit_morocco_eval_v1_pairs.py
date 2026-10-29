"""Montage of closest pHash pairs (eval-internal, eval vs AIHub) + Ultralytics yaml check."""
from pathlib import Path

from ultralytics.data.utils import check_det_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(r"C:\workspace\gpr_detection")
EV = ROOT / "data" / "morocco_eval_v1" / "images"
AD = ROOT / "data" / "aihub_yolo_v1" / "images"
OUT = ROOT / "analysis" / "data" / "out"
internal = pd.read_csv(OUT / "morocco_eval_phash_internal_le8.csv")
near = pd.read_csv(OUT / "morocco_eval_vs_aihub_nearest.csv").nsmallest(5, "min_dist")
rows = [(EV / r.a, EV / r.b, f"internal d={r.dist}") for r in internal.itertuples()]
rows += [(EV / r.eval, AD / r.aihub, f"vs AIHub d={r.min_dist}") for r in near.itertuples()]
fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)))
for i, (a, b, t) in enumerate(rows):
    for j, p in enumerate((a, b)):
        axes[i, j].imshow(Image.open(p).convert("L"), cmap="gray")
        axes[i, j].set_title(p.name if j == 0 else f"{p.name}\n{t}", fontsize=6)
        axes[i, j].axis("off")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "datasets" / "figures" / "morocco_eval_v1_phash_pairs_montage.png", dpi=100)

