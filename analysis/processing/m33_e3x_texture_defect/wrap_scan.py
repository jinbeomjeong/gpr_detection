"""Physics side checks of the E3x time-jitter arm on every pool line (CPU, blind).

Per texture position (same arms and call order as the sealed runner, reused from
python-expert's diagnostic ``texture_scan.py``) this records, for sigma 0.1/0.2 P:

* b2 identity residual: max_x |E_x/E - 1 + r_N sin^2(pi delta_x)|,
* circular-wrap effect: sealed circular jitter vs a linear (zero-padded 4n)
  jitter of the same column, max |diff| / max |column| (amplitude ratio) and
  the energy of the difference above the truth top row0 (relative to E),
* fix (a): max |band-limited column - column| / max |column| and the same
  restricted to the truth rows [row0, row1],
* column edge content: max |column| in the first / last ceil(max|delta|)+1 rows
  relative to max |column|.

No detection, image, recall or score is built. Outputs: one csv + summary json.
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

DIAG = Path(r"C:\Users\jinbeom\.claude\runners\gpr_detection-m33_e3x\diag_texture_20261120")
sys.path.insert(0, str(DIAG))

import numpy as np  # noqa: E402

import texture_scan as ts  # noqa: E402

e3x = ts.e3x


def linear_jitter(column: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Same phase rotation on a 4n zero-padded column (no wrap), cropped to n."""
    n = column.shape[0]
    padded = np.zeros(4 * n)
    padded[n : 2 * n] = column
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(4 * n)
    phases = np.exp(-2j * np.pi * np.outer(freqs, delays))
    out = np.fft.irfft(spectrum[:, None] * phases, n=4 * n, axis=0)
    return out[n : 2 * n]


def position_row(line_id, key, built, textures, period):
    band = built.band
    w, _m = textures.fields[key]
    column = np.asarray(e3x.place.apex_column(band), dtype=np.float64)
    n = column.shape[0]
    peak = float(np.max(np.abs(column)))
    energy = float(np.sum(column**2))
    nyq, above = ts.spectral_fractions(column)
    r_n = 0.0 if nyq is None else nyq
    limited = ts.band_limited(column)
    row0, row1 = int(band.row0), int(min(band.row1, n - 1))
    row = {
        "line_id": line_id,
        "position_id": key,
        "n_rows": n,
        "row0": row0,
        "nyq_frac": r_n,
        "above_frac": above,
        "fix_a_maxdiff_rel": float(np.max(np.abs(limited - column)) / peak),
        "fix_a_maxdiff_truth_rel": float(
            np.max(np.abs(limited[row0 : row1 + 1] - column[row0 : row1 + 1])) / peak
        ),
    }
    for sigma in ts.SIGMAS:
        tag = f"{sigma:g}"
        delays = e3x.place.jitter_delays(w, sigma, period)
        circ = e3x.place.time_jitter(column, w, sigma, period)
        lin = linear_jitter(column, delays)
        e_x = np.sum(circ**2, axis=0) / energy
        resid = np.abs(e_x - 1.0 + r_n * np.sin(np.pi * delays) ** 2)
        diff = circ - lin
        edge = math.ceil(float(np.max(np.abs(delays)))) + 1
        row |= {
            f"max_abs_delay_samples_{tag}": float(np.max(np.abs(delays))),
            f"max_abs_delay_P_{tag}": float(np.max(np.abs(delays)) / period),
            f"b2_resid_{tag}": float(np.max(resid)),
            f"energy_err_{tag}": float(np.max(np.abs(e_x - 1.0))),
            f"wrap_maxdiff_rel_{tag}": float(np.max(np.abs(diff)) / peak),
            f"wrap_energy_above_row0_{tag}": float(
                np.max(np.sum(diff[:row0] ** 2, axis=0)) / energy
            )
            if row0 > 0
            else 0.0,
            f"edge_top_rel_{tag}": float(np.max(np.abs(column[:edge])) / peak),
            f"edge_bottom_rel_{tag}": float(np.max(np.abs(column[-edge:])) / peak),
        }
    return row


def scan_line(plan, band_library):
    line_id = plan.line.line_id
    line = e3x.runner.load_line(plan.line.ref)
    context = e3x.inject.prepare_line_context(line.radargram, plan.exclusions)
    responses = e3x.line_responses(band_library, line, e3x.BAND_MODELS)
    built = {}
    for arm in plan.arms:
        band = e3x.rebuild_arm(context, responses[arm.model], arm)
        if band is not None:
            built.setdefault((arm.row_px, arm.arm), []).append(e3x.BuiltArm(arm, band))
    textures = e3x._Textures()
    first_built = {}
    for (row_px, arm_name), arms in sorted(built.items()):
        arms = sorted(arms, key=lambda a: a.planned.position_id)
        for texture, _snr, _pol in e3x.band_conditions(arm_name, row_px):
            if texture == e3x.TEXTURE_NONE:
                continue
            for arm in arms:
                first_built.setdefault(arm.planned.position_id, arm)
                textures.of(arm, context.period_traces, context.period_samples)
    return [
        position_row(line_id, k, first_built[k], textures, context.period_samples)
        for k in sorted(textures.checks)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    library = e3x.responses_io.load_library()
    plans = ts.chosen_plans("all")
    rows, errors = [], []
    for plan in plans:
        try:
            rows += scan_line(plan, library)
        except Exception as error:  # a scan, not a judgment
            errors.append(f"{plan.line.line_id}: {type(error).__name__}: {error}")
    keys = [k for k in rows[0] if k not in ("line_id", "position_id")]
    summary = {
        "n_lines": len(plans),
        "n_positions": len(rows),
        "errors": errors,
        "max": {k: max(float(r[k]) for r in rows) for k in keys},
        "seconds": round(time.perf_counter() - started, 1),
    }
    with (args.out_dir / "positions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps({"status": "ok", "n_positions": len(rows), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
