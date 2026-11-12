"""M2-10 (c) P3 flag behaviour on the gprMax lines (report only, no verdict).

docs/processing/m210_c_qc.md v1.2 sections 4-2 and 5 (C-7). Read only: the
exported effect lines (``data/synthetic_gprmax_c_qc``, ``--c-qc`` export) and the
reused K7 controls (``data/synthetic_gprmax_k7``). No depth is computed.

Per model (m = 0 Z_HSTEP, 1 Z_HWAVE, 2 K7_P1_D060, 3 K7_P2_D070_L): the line is
resampled to 0.098 ns (the K7 harness rule) and processed with
``DepthConfig(band_shift_qc=True)``; one noise-free line, then rho {10, 20} x
seed 0-19 with white noise ``sigma_raw = A0 / (rho sigma_u)`` (K7 rule: ``A0`` =
smallest pipe apex envelope, ``sigma_u`` from the unit-noise section) and seed
``20311112 + 1000 m + 100 q + s``. For every line: ``sigma_s``, ``C0``, ``C1``,
``band_shift_n_valid`` and the three P3 flags; on the effect lines also the
slope of the measured shifts ``s~_j`` on the json expected shifts (geometric
``expected_band_shift_ns`` and the library band peaks ``expected_band_peak_ns``
minus their mean), the "follow ratio".

Outputs (LF, project-relative paths): ``docs/processing/m210c_flag_report.csv``
(one row per line) and ``docs/processing/m210c_flag_report.json`` (per model and
rho: quantiles, flag rates, slopes; seed overlap with every earlier seed set).

Run (one core, BelowNormal):
    conda run -n gpr-dev python -I analysis/processing/m210c_flag_report.py
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Reproducible MKL results across processes, before numpy loads (M2-10).
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import m210_k7_pipeline as k7  # noqa: E402
import m210b_floor_rules as floor_rules  # noqa: E402
import m210b_s4s6_pipeline as floor_harness  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.io.synthetic import (  # noqa: E402
    load_synthetic,
    load_synthetic_pipe_truths,
)
from gpr_detection.preprocess.timezero import (  # noqa: E402
    ZERO_SHIFT_LARGE,
    ZERO_SHIFT_NOT_SURFACE,
    ZERO_TRACKS_SURFACE,
    band_shift_qc,
)

SPEC = "docs/processing/m210_c_qc.md v1.2 sections 4-2, 5 (C-7); report only"
EFFECT_MODELS = ("Z_HSTEP", "Z_HWAVE")
CONTROL_MODELS = ("K7_P1_D060", "K7_P2_D070_L")
MODEL_IDS = (*EFFECT_MODELS, *CONTROL_MODELS)
"""Seed model index m (section 4-2)."""
RHOS = (10, 20)
N_SEEDS = 20
SEED_BASE = 20311112
UNIT_NOISE_SEED = 20311199
"""Unit-noise section seed (the v1.0 value; v1.1 names none)."""
A_ANALYSIS_SEEDS = frozenset({20261121, 20261122, 20261123})
CONFIG = DepthConfig(band_shift_qc=True)
P3_FLAGS = (ZERO_SHIFT_LARGE, ZERO_TRACKS_SURFACE, ZERO_SHIFT_NOT_SURFACE)
DEFAULT_EFFECT_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_c_qc"
DEFAULT_CONTROL_ROOT = PROJECT_ROOT / "data" / "synthetic_gprmax_k7"
DEFAULT_OUT_JSON = PROJECT_ROOT / "docs" / "processing" / "m210c_flag_report.json"
DEFAULT_OUT_CSV = PROJECT_ROOT / "docs" / "processing" / "m210c_flag_report.csv"
QUANTILES = (0.1, 0.5, 0.9)
ROW_COLUMNS = (
    "model_id",
    "role",
    "rho",
    "seed",
    "seed_index",
    "noise_free",
    "t_zero_ns",
    "zero_rule",
    "band_shift_std_ns",
    "band_c0",
    "band_c1",
    "band_c1_minus_c0",
    "band_shift_n_valid",
    *P3_FLAGS,
    "follow_slope_geometry",
    "follow_slope_band_peak",
)


# ----------------------------------------------------------------- seeds


def noise_seed(model_index: int, rho_index: int, seed_index: int) -> int:
    """``20311112 + 1000 m + 100 q + s`` (section 4-2)."""
    return SEED_BASE + 1000 * model_index + 100 * rho_index + seed_index


def report_seeds() -> set[int]:
    seeds = {
        noise_seed(m, q, s)
        for m in range(len(MODEL_IDS))
        for q in range(len(RHOS))
        for s in range(N_SEEDS)
    }
    return seeds | {UNIT_NOISE_SEED}


def seed_record() -> dict[str, object]:
    """Overlap of the report seeds with every earlier seed set (recorded, 0 expected).

    Raises:
        floor_harness.SeedOverlapError: A shared seed.
    """
    earlier = {
        **k7.earlier_seed_sets(),
        "k7_harness": frozenset(k7.k7_seeds()),
        "a_eps_bias": A_ANALYSIS_SEEDS,
    }
    seeds = report_seeds()
    overlap = floor_harness.assert_no_seed_overlap(
        seeds, set(frozenset().union(*earlier.values()))
    )
    return {
        "seed_overlap": overlap,
        "checked_against": {name: len(values) for name, values in earlier.items()},
        "seed_formula": f"{SEED_BASE} + 1000 m + 100 q + s, s 0-{N_SEEDS - 1}",
        "unit_noise_seed": UNIT_NOISE_SEED,
        "n_seeds": len(seeds),
    }


# ----------------------------------------------------------------- models


@dataclass(frozen=True)
class ReportModel:
    """Resampled noise-free line of one model and its noise levels."""

    model_id: str
    role: str
    radargram: Radargram
    sigma_raw_by_rho: Mapping[int, float]
    expected_geometry_ns: np.ndarray | None
    expected_band_peak_ns: np.ndarray | None


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return path.as_posix()


def _expected_shifts(meta: Mapping) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Geometric and library-peak expected band shifts (mean removed), or None."""
    c_qc = meta.get("c_qc")
    if c_qc is None:
        return None, None
    geometry = np.asarray(c_qc["expected_band_shift_ns"], dtype=np.float64)
    peaks = c_qc.get("expected_band_peak_ns")
    if peaks is None:
        return geometry, None
    peaks = np.asarray(peaks, dtype=np.float64)
    return geometry, peaks - float(np.mean(peaks))


def load_model(model_id: str, root: Path) -> ReportModel:
    """Exported line, resampled, with its K7-rule noise levels."""
    radargram = load_synthetic(
        model_id, root=root, include_eps=True, include_radius=False,
        include_height=True,
    )  # fmt: skip
    truths = load_synthetic_pipe_truths(model_id, root=root)
    meta = json.loads((root / f"{model_id}.json").read_text(encoding="utf-8"))
    resampled = floor_harness.resample_to(radargram, floor_harness.DT_HARNESS_NS)
    clean = k7.clean_model(model_id, resampled, truths)
    if clean.t_zero_ns is None or clean.a0 is None:
        raise ValueError(f"{model_id}: no zero or no pipe to calibrate the noise")
    k_lo, k_hi = floor_rules.search_rows(
        clean.t_zero_ns, resampled.f_c_hz, resampled.dt_ns, resampled.n_samples
    )
    sigma_u = floor_harness.unit_noise_sigma(resampled, k_lo, k_hi, UNIT_NOISE_SEED)
    geometry, band_peak = _expected_shifts(meta)
    return ReportModel(
        model_id=model_id,
        role="effect" if model_id in EFFECT_MODELS else "control",
        radargram=resampled,
        sigma_raw_by_rho={rho: clean.a0 / (rho * sigma_u) for rho in RHOS},
        expected_geometry_ns=geometry,
        expected_band_peak_ns=band_peak,
    )


# ----------------------------------------------------------------- one line


def follow_slope(shifts_ns: np.ndarray, expected_ns: np.ndarray | None) -> float | None:
    """Least-squares slope of the measured shifts on the expected ones."""
    if expected_ns is None:
        return None
    valid = np.isfinite(shifts_ns) & np.isfinite(expected_ns)
    if valid.sum() < 2 or np.ptp(expected_ns[valid]) == 0:
        return None
    return float(np.polyfit(expected_ns[valid], shifts_ns[valid], 1)[0])


def _rounded(value: float | None, digits: int = 6) -> object:
    return (
        "" if value is None or not np.isfinite(value) else round(float(value), digits)
    )


def line_row(
    model: ReportModel,
    line: Radargram,
    rho: int | None,
    seed: int | None,
    seed_index: int | None,
) -> dict[str, object]:
    """Band-shift QC of one (noisy) line; no depth."""
    inputs = prepare_depth_inputs(line, CONFIG)
    zero = inputs.zero
    shifts = None
    if zero.t_zero_ns is not None and zero.band_shift_std_ns is not None:
        shifts = band_shift_qc(
            inputs.branch.filtered, line.dt_ns, line.f_c_hz, zero.t_zero_ns,
            np.ones(line.n_traces, dtype=bool),
        ).shifts_ns  # fmt: skip
    c0, c1 = zero.band_coherence or (None, None)
    row = {
        "model_id": model.model_id,
        "role": model.role,
        "rho": "" if rho is None else rho,
        "seed": "" if seed is None else seed,
        "seed_index": "" if seed_index is None else seed_index,
        "noise_free": rho is None,
        "t_zero_ns": _rounded(zero.t_zero_ns, 4),
        "zero_rule": str(zero.zero_rule),
        "band_shift_std_ns": _rounded(zero.band_shift_std_ns),
        "band_c0": _rounded(c0),
        "band_c1": _rounded(c1),
        "band_c1_minus_c0": "" if c0 is None or c1 is None else _rounded(c1 - c0),
        "band_shift_n_valid": ""
        if zero.band_shift_n_valid is None
        else zero.band_shift_n_valid,
        "follow_slope_geometry": "",
        "follow_slope_band_peak": "",
    }
    row.update({flag: flag in zero.flags for flag in P3_FLAGS})
    if shifts is not None:
        row["follow_slope_geometry"] = _rounded(
            follow_slope(shifts, model.expected_geometry_ns), 4
        )
        row["follow_slope_band_peak"] = _rounded(
            follow_slope(shifts, model.expected_band_peak_ns), 4
        )
    return {column: row[column] for column in ROW_COLUMNS}


def model_rows(model: ReportModel) -> list[dict[str, object]]:
    """Noise-free line, then rho x seed lines of one model."""
    model_index = MODEL_IDS.index(model.model_id)
    rows = [line_row(model, model.radargram, None, None, None)]
    shape = model.radargram.data.shape
    for rho_index, rho in enumerate(RHOS):
        for seed_index in range(N_SEEDS):
            seed = noise_seed(model_index, rho_index, seed_index)
            noise = model.sigma_raw_by_rho[rho] * floor_harness.white_noise(shape, seed)
            noisy = floor_harness.with_noise(model.radargram, noise)
            rows.append(line_row(model, noisy, rho, seed, seed_index))
    return rows


# ----------------------------------------------------------------- summary


def _quantiles(values: Sequence[object]) -> list[float] | None:
    numbers = [float(v) for v in values if v != ""]
    if not numbers:
        return None
    return [round(float(np.quantile(numbers, q)), 6) for q in QUANTILES]


def summarise(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Per model and rho (``noise_free`` for the clean line): quantiles and rates."""
    summary: dict[str, object] = {}
    for model_id in MODEL_IDS:
        for rho in ("noise_free", *RHOS):
            selected = [
                row for row in rows
                if row["model_id"] == model_id
                and (row["noise_free"] if rho == "noise_free" else row["rho"] == rho)
            ]  # fmt: skip
            if not selected:
                continue
            summary[f"{model_id}|{rho}"] = {
                "n_lines": len(selected),
                "band_shift_std_ns_q10_q50_q90": _quantiles(
                    [r["band_shift_std_ns"] for r in selected]
                ),
                "band_c1_minus_c0_q10_q50_q90": _quantiles(
                    [r["band_c1_minus_c0"] for r in selected]
                ),
                "band_shift_n_valid_q10_q50_q90": _quantiles(
                    [r["band_shift_n_valid"] for r in selected]
                ),
                "flag_rates": {
                    flag: sum(bool(r[flag]) for r in selected) / len(selected)
                    for flag in P3_FLAGS
                },
                "follow_slope_geometry_q10_q50_q90": _quantiles(
                    [r["follow_slope_geometry"] for r in selected]
                ),
                "follow_slope_band_peak_q10_q50_q90": _quantiles(
                    [r["follow_slope_band_peak"] for r in selected]
                ),
            }
    return summary


# ----------------------------------------------------------------- entry point


def input_files(effect_root: Path, control_root: Path) -> dict[str, dict]:
    """SHA-256 of every line file read (project-relative paths)."""
    files = {}
    for model_id in MODEL_IDS:
        root = effect_root if model_id in EFFECT_MODELS else control_root
        for suffix in (".npy", ".json"):
            path = root / f"{model_id}{suffix}"
            files[_relative(path)] = floor_harness._sha256(path)
    return files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--effect-root", type=Path, default=DEFAULT_EFFECT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = seed_record()
    rows: list[dict[str, object]] = []
    noise_levels = {}
    for model_id in MODEL_IDS:
        root = args.effect_root if model_id in EFFECT_MODELS else args.control_root
        model = load_model(model_id, root)
        noise_levels[model_id] = {str(k): v for k, v in model.sigma_raw_by_rho.items()}
        rows += model_rows(model)
        print(f"{model_id}: {len(rows)} lines so far", flush=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ROW_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "spec": SPEC,
        "note": "report only: no threshold, no verdict, no depth (C-7)",
        "seeds": seeds,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "inputs": input_files(args.effect_root, args.control_root),
        "noise_sigma_raw_by_rho": noise_levels,
        "rows_csv": _relative(args.out_csv),
        "rows_csv_sha256": floor_harness._sha256(args.out_csv),
        "summary": summarise(rows),
    }
    args.out_json.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {_relative(args.out_csv)} and {_relative(args.out_json)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
