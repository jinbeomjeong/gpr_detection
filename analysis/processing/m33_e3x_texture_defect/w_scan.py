"""Shape of the sealed texture field w per E3x texture position (CPU, blind).

Records the flat length against the kernel SD, the w range on the flat, box and
taper columns, and the realized delay (in P) at sigma 0.2 on those columns.
Also the ensemble SD of the unnormalized smoothed noise (analytic, kernel
norm) against the per-position flat SD, i.e. the gain the flat normalization
applies. No detection, image or score is built.
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

DIAG = Path(r"C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120")
sys.path.insert(0, str(DIAG))

import numpy as np  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

import texture_scan as ts  # noqa: E402

e3x = ts.e3x
place = e3x.place


def smoothed_raw(n_columns, period_traces, key):
    """The sealed texture_field before normalization (same rng draw)."""
    sigma = place.TEXTURE_CORRELATION_PERIODS * period_traces
    half = math.ceil(place.KERNEL_HALF_WIDTH_SIGMAS * sigma)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    noise = place.texture_rng(key).standard_normal(n_columns)
    return fftconvolve(noise, kernel, mode="same"), float(np.sqrt(np.sum(kernel**2))), sigma


def position_row(line_id, key, built, textures, context):
    band = built.band
    w, _m = textures.fields[key]
    raw, ens_sd, sigma_tr = smoothed_raw(band.n_columns, context.period_traces, key)
    first, last = band.flat_columns
    b0, b1 = band.box_columns
    flat = slice(first, last + 1)
    w_check = (raw - raw[flat].mean()) / raw[flat].std()
    taper_cols = np.r_[0:first, last + 1 : band.n_columns]
    return {
        "line_id": line_id,
        "position_id": key,
        "w_reproduced_maxdiff": float(np.max(np.abs(w_check - w))),
        "n_columns": band.n_columns,
        "taper_traces": band.taper_traces,
        "n_flat": last - first + 1,
        "kernel_sd_traces": sigma_tr,
        "flat_over_kernel_sd": (last - first + 1) / sigma_tr,
        "flat_sd_over_ensemble_sd": float(raw[flat].std() / ens_sd),
        "w_absmax_flat": float(np.max(np.abs(w[flat]))),
        "w_absmax_box": float(np.max(np.abs(w[b0 : b1 + 1]))),
        "w_absmax_taper": float(np.max(np.abs(w[taper_cols]))) if taper_cols.size else 0.0,
        "w_range_flat": float(np.ptp(w[flat])),
        "delay02_absmax_box_P": 0.2 * float(np.max(np.abs(w[b0 : b1 + 1]))),
        "delay02_sd_box_P": 0.2 * float(np.std(w[b0 : b1 + 1])),
        "delay02_sd_box_detrended_P": 0.2
        * float(
            np.std(
                w[b0 : b1 + 1]
                - np.polyval(
                    np.polyfit(np.arange(b1 - b0 + 1), w[b0 : b1 + 1], 1),
                    np.arange(b1 - b0 + 1),
                )
            )
        ),
    }


def scan_line(plan, library):
    line_id = plan.line.line_id
    line = e3x.runner.load_line(plan.line.ref)
    context = e3x.inject.prepare_line_context(line.radargram, plan.exclusions)
    responses = e3x.line_responses(library, line, e3x.BAND_MODELS)
    built = {}
    for arm in plan.arms:
        band = e3x.rebuild_arm(context, responses[arm.model], arm)
        if band is not None:
            built.setdefault((arm.row_px, arm.arm), []).append(e3x.BuiltArm(arm, band))
    textures = e3x._Textures()
    first_built = {}
    for (row_px, arm_name), arms in sorted(built.items()):
        arms = sorted(arms, key=lambda a: a.planned.position_id)
        for texture, _s, _p in e3x.band_conditions(arm_name, row_px):
            if texture == e3x.TEXTURE_NONE:
                continue
            for arm in arms:
                first_built.setdefault(arm.planned.position_id, arm)
                textures.of(arm, context.period_traces, context.period_samples)
    return [position_row(line_id, k, first_built[k], textures, context) for k in sorted(textures.checks)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    library = e3x.responses_io.load_library()
    rows = []
    for plan in ts.chosen_plans("all"):
        rows += scan_line(plan, library)
    keys = [k for k in rows[0] if k not in ("line_id", "position_id")]
    summary = {"n_positions": len(rows)}
    for k in keys:
        values = np.array([float(r[k]) for r in rows])
        summary[k] = {
            "min": float(values.min()),
            "p50": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        }
    with (args.out_dir / "w_positions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "w_summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps({"status": "ok", "n_positions": len(rows)}))


if __name__ == "__main__":
    main()
