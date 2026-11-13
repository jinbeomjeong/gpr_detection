"""Where does the L1 period differ from P*? Whole content vs a window at the pipe apex (T4 review).

Synthetic: tests/synthetic_bscan.py line of T4 (500 MHz, dt 0.098, 401 traces), noise-free and
SNR 20 dB. Real: 4TU true apexes of the blinded review on 01.1/Path1 (G114) and 02.x lines.
L1 = make_detection_image (implemented to_image.py, read-only use). Apex window in L1:
rows apex-2 P* .. apex+4 P*, columns +-7 around the apex column; the AGC'd gray image is
measured with the same estimator (measure_period_samples) after removing the 127.5 offset.
Also measures the same window and the whole content on a no-AGC variant (resample only).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

ROOT = Path(r"C:\workspace\gpr_detection")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.to_image import (  # noqa: E402
    DetectionImageConfig,
    make_detection_image,
    measure_period_samples,
)
from synthetic_bscan import BScanSpec, make_radargram  # noqa: E402

FOURTU = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")


def window_period(img, row, col, p_star):
    r0, r1 = max(0, int(row - 2 * p_star)), int(row + 4 * p_star) + 1
    c0, c1 = max(0, int(col) - 7), int(col) + 8
    win = img[r0:r1, c0:c1].astype(np.float64) - 127.5
    if win.shape[0] < 8 or win.shape[1] < 1:
        return None
    return measure_period_samples(win, min_period=2.0)


def report(name, radargram, apex_trace, apex_sample, p_star):
    image = make_detection_image(radargram, DetectionImageConfig(p_target_px=p_star))
    content = image.line_image[: image.content_height_px]
    gm = image.grid_map
    row = apex_sample / gm.samples_per_px
    col = apex_trace / gm.traces_per_px
    whole = measure_period_samples(content.astype(np.float64) - 127.5, min_period=2.0)
    local = window_period(image.line_image, row, col, p_star)
    print(f"{name} P*={p_star}: P_l1={image.period_l1_px:.2f} whole={whole} apex_window={local}")


def synthetic_apex(radargram):
    br = prepare_fitting_branch(radargram).background_removed
    centre = br.shape[1] // 2
    env = np.abs(hilbert(br[:, centre]))
    env[:60] = 0
    return centre, int(np.argmax(env))


for snr in (None, 20.0):
    spec = BScanSpec(f_c_hz=500e6, dt_ns=0.098, n_traces=401, n_samples=512, air_gap_m=0, snr_db=snr)
    rg = make_radargram(spec)
    tr, k = synthetic_apex(rg)
    for p_star in (4.06, 6.0):
        report(f"synthetic snr={snr} apex=({tr},{k})", rg, tr, k, p_star)

rows = [r for r in csv.DictReader(open(ROOT / "docs" / "processing" / "m29c_apex_review.csv", encoding="utf-8-sig"))
        if r["type"] == "T"][:6]
for r in rows:
    rg = load_fourtu_radargram(FOURTU, r["activity_id"], r["path"])
    report(f"4TU {r['activity_id']}/{r['path']} apex=({r['trace']},{r['sample']})", rg,
           int(r["trace"]), float(r["sample"]), 4.06)
