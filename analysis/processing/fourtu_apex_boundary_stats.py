"""Box-top boundary statistics of the M2-8 4TU apexes (spec_depth_v1 5-7 evidence).

Reads only the committed csvs (no radargram, no npy):
``docs/processing/depth_4tu_v1_candidates.csv`` (M2-8 run, code before the
section 5-7 fix) and ``docs/processing/fourtu_lines_v1.csv`` (``dt_ns``).

The M2-8 candidates csv has no box columns, so the search top is recomputed
exactly as ``scripts/eval_depth_4tu.py`` and ``extract_picks`` did:

* box top (``candidate_box``, widening changes traces only):
  ``k0 = max(0, floor(candidate_sample - 1.5 * T / dt))``
* search top after the mute (``mute_sample``):
  ``k_top = max(k0, ceil((t_zero_ns + T) / dt))``

with ``T = 1 / f_c = 2 ns`` (4TU antenna 500 MHz, ``io/fourtu.py``). The
distance of an ``ok`` apex from the search top is ``apex_sample - k_top``
(samples); 0-2 samples is the box-top boundary effect.

Prints the statistics; ``--out`` also writes them as a two-column csv
(``metric, value``).

Run:  conda run -n gpr-dev python -I analysis/processing/fourtu_apex_boundary_stats.py
      [--out docs/processing/depth_4tu_v1_boundary_stats.csv]
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSING = Path(__file__).resolve().parents[2] / "docs" / "processing"
CANDIDATES_CSV = PROCESSING / "depth_4tu_v1_candidates.csv"
LINES_CSV = PROCESSING / "fourtu_lines_v1.csv"

PERIOD_NS = 2.0
"""1 / f_c of the 4TU antenna (500 MHz)."""
BOX_MARGIN_PERIODS = 1.5
BOUNDARY_SAMPLES = 2
QUANTILES_PCT = (0, 5, 25, 50, 75, 95, 100)
HISTOGRAM_EDGES = (0, 1, 2, 3, 5, 10, 20, 40, 61)
"""Distance bins in samples; distances above 60 go into the last bin."""
EARLY_SHIFT_NS = -2.5


def load_ok_apexes() -> pd.DataFrame:
    """``ok`` rows with ``dt_ns``, the recomputed search top and the distances."""
    candidates = pd.read_csv(CANDIDATES_CSV, dtype={"activity_id": str})
    lines = pd.read_csv(LINES_CSV, dtype={"activity_id": str})
    ok = candidates[candidates["status"] == "ok"].merge(
        lines[["activity_id", "path", "dt_ns"]],
        on=["activity_id", "path"],
        how="left",
        validate="many_to_one",
    )
    if ok["dt_ns"].isna().any():
        raise ValueError("ok candidates without a line dt_ns")
    margin = BOX_MARGIN_PERIODS * PERIOD_NS
    ok["box_k0"] = [
        max(0, math.floor(sample - margin / dt))
        for sample, dt in zip(ok["candidate_sample"], ok["dt_ns"], strict=True)
    ]
    ok["k_mute"] = [
        math.ceil((t_zero + PERIOD_NS) / dt)
        for t_zero, dt in zip(ok["t_zero_ns"], ok["dt_ns"], strict=True)
    ]
    ok["search_k0"] = ok[["box_k0", "k_mute"]].max(axis=1)
    ok["distance_samples"] = ok["apex_sample"] - ok["search_k0"]
    ok["distance_ns"] = ok["distance_samples"] * ok["dt_ns"]
    ok["shift_ns"] = (ok["apex_sample"] - ok["candidate_sample"]) * ok["dt_ns"]
    return ok


def _boundary_fraction(distance_samples: pd.Series) -> float:
    inside = (distance_samples >= 0) & (distance_samples <= BOUNDARY_SAMPLES)
    return float(inside.mean())


def boundary_statistics(ok: pd.DataFrame) -> dict[str, float | int]:
    """Flat ``metric -> value`` table of the boundary statistics."""
    distance = ok["distance_samples"]
    stats: dict[str, float | int] = {"n_ok": len(ok)}
    for pct, value in zip(
        QUANTILES_PCT, np.percentile(distance, QUANTILES_PCT), strict=True
    ):
        stats[f"distance_samples_q{pct}"] = round(float(value), 2)
    stats["frac_distance_0_to_2_samples"] = round(_boundary_fraction(distance), 4)
    stats["frac_distance_le_0.25_period"] = round(
        float((ok["distance_ns"] <= 0.25 * PERIOD_NS).mean()), 4
    )
    stats["frac_distance_le_0.5_period"] = round(
        float((ok["distance_ns"] <= 0.5 * PERIOD_NS).mean()), 4
    )
    stats["median_shift_ns"] = round(float(ok["shift_ns"].median()), 3)
    stats["frac_shift_le_-2.5ns"] = round(
        float((ok["shift_ns"] <= EARLY_SHIFT_NS).mean()), 4
    )
    stats["frac_mute_clipped"] = round(float((ok["k_mute"] > ok["box_k0"]).mean()), 4)
    counts, _ = np.histogram(distance.clip(upper=60), bins=HISTOGRAM_EDGES)
    for low, high, count in zip(
        HISTOGRAM_EDGES[:-1], HISTOGRAM_EDGES[1:], counts, strict=True
    ):
        stats[f"hist_samples_{low}_to_{high}"] = int(count)
    for dt, group in ok.groupby("dt_ns"):
        stats[f"dt_{dt}_n"] = len(group)
        stats[f"dt_{dt}_frac_0_to_2"] = round(
            _boundary_fraction(group["distance_samples"]), 4
        )
    for site, group in ok.groupby("site"):
        stats[f"site_{site}_n"] = len(group)
        stats[f"site_{site}_frac_0_to_2"] = round(
            _boundary_fraction(group["distance_samples"]), 3
        )
    return stats


def write_stats(stats: dict[str, float | int], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(stats.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="optional csv")
    args = parser.parse_args()
    stats = boundary_statistics(load_ok_apexes())
    for metric, value in stats.items():
        print(f"{metric}: {value}")
    if args.out is not None:
        write_stats(stats, args.out)


if __name__ == "__main__":
    main()
