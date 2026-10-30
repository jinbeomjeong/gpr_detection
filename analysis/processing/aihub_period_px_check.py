"""Why is the AIHub no-scale dominant period ~4 px? (band spec v1.1 follow-up)

For a few train_pool gain-1.0 crops with ``pipe_band`` (same seeded sample as
scripts/check_band_picks_aihub.py, test split untouched) this prints:

* vertical (time-axis) trace-averaged |rfft| peaks of the whole crop,
* the same inside the band boxes only (box rows +-8 px, box columns),
* horizontal (trace-axis) spectrum at 1/8, 1/4, 3/8 cycles/px vs neighbours
  (JPEG 8x8 block artifacts show up in both directions; a radar carrier only
  vertically),
* the strongest vertical peak restricted to 8-128 px.

Read-only; images from data/aihub_yolo_v1 (converted copies). No file names or
box coordinates are printed (AIHub card rule). Single process, a few seconds.
Run:  conda run -n gpr-dev python -I analysis/processing/aihub_period_px_check.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_band_picks_aihub import DATASET_DIR, band_boxes, sample_images  # noqa: E402
from gpr_detection.io._common import read_grayscale_file  # noqa: E402

N_IMAGES = 10


def vertical_spectrum(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = block.astype(np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(values, axis=0)).mean(axis=1)
    return np.fft.rfftfreq(values.shape[0], d=1.0), spec


def top_periods(freq, spec, k=3, lo_px=2.0, hi_px=256.0):
    sel = (freq > 0) & (1 / np.maximum(freq, 1e-9) >= lo_px) & (1 / np.maximum(freq, 1e-9) <= hi_px)
    idx = np.flatnonzero(sel)
    order = idx[np.argsort(spec[idx])[::-1][:k]]
    total = spec[sel].sum()
    return ", ".join(f"{1 / freq[i]:.2f}px({spec[i] / total:.1%})" for i in order)


def block_line_ratio(img: np.ndarray, axis: int) -> str:
    values = img.astype(np.float64)
    values = values - values.mean(axis=axis, keepdims=True)
    spec = np.abs(np.fft.rfft(values, axis=axis)).mean(axis=1 - axis)
    freq = np.fft.rfftfreq(values.shape[axis], d=1.0)
    out = []
    for f0 in (0.125, 0.25, 0.375):
        i = int(np.argmin(np.abs(freq - f0)))
        neighbours = np.r_[spec[i - 4:i - 1], spec[i + 2:i + 5]].mean()
        out.append(f"{f0:.3f}:{spec[i] / neighbours:.2f}")
    return " ".join(out)


def main() -> None:
    names = sample_images(40)[:N_IMAGES]
    for n, name in enumerate(names):
        img = read_grayscale_file(DATASET_DIR / "images" / name)
        h, w = img.shape
        freq, spec = vertical_spectrum(img)
        boxes = band_boxes(DATASET_DIR / "labels" / f"{Path(name).stem}.txt", w, h)
        print(f"[img {n}] {h}x{w}, band boxes {len(boxes)}")
        print(f"   whole vertical top: {top_periods(freq, spec)}")
        print(f"   whole vertical top in 8-128 px: {top_periods(freq, spec, k=2, lo_px=8, hi_px=128)}")
        print(f"   block-line ratio vertical  {block_line_ratio(img, 0)}")
        print(f"   block-line ratio horizontal {block_line_ratio(img, 1)}")
        for b in boxes[:2]:
            k0, k1 = max(0, b.k0 - 8), min(h, b.k1 + 9)
            sub = img[k0:k1, b.i0:b.i1 + 1]
            if sub.shape[0] < 16:
                print(f"   box h {b.k1 - b.k0 + 1}px: too short for a spectrum")
                continue
            # zero-pad to 256 for a finer frequency grid
            pad = np.zeros((256, sub.shape[1]))
            pad[: sub.shape[0]] = sub - sub.mean(axis=0, keepdims=True)
            bf = np.fft.rfftfreq(256, d=1.0)
            bs = np.abs(np.fft.rfft(pad, axis=0)).mean(axis=1)
            print(f"   box h {b.k1 - b.k0 + 1}px w {b.i1 - b.i0 + 1}px: vertical top {top_periods(bf, bs)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
