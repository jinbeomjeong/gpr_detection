"""Tile counts of the C' detection image, [SP] v0.3 formulas.

[SP] = docs/processing/spec_detection_image_v1.md v0.3, §4-4 3 (scale and
compression), §4-4 9 (overlap) and §4-4 10 (tile layout):
    s_y = P_line / P*,  s_x = s_y * delta_z / dx   (delta_z = v dt / 2)
    if round(n_s / s_y) > L: c = (n_s / L) / s_y, both scales *= c
        (period_compressed; scale_out_of_range when c > 1.25)
    H_c = round(n_s / s_y), W1 = round(N / s_x), sigma_x = N / W1
    O_phys = max(64, ceil(min(2 a_max / (sigma_x dx), 20 P*) - 1e-9))
        (overlap_period_capped when the period term wins)
    O = min(224, max(O_phys, O_cv)),  tiles = 1 + ceil((W1 - L) / (L - O)) if W1 > L

Period and velocity per line are assumptions close to the [SP] defaults:
  - 4TU: P = 2.51 ns for every line, v from eps_r_used (metadata, not measured).
  - Guangzhou pipe: 400 MHz 30 ns window 2.93 ns, 50 ns window 2.11 ns
    (measured medians, [SP] §2-2), other antennas 1000 / f_MHz ns, MALA 2.5 ns
    ([SP] DATASET_DEFAULT_PERIOD); v = 0.10 m/ns. [SP] measures P per line, so
    real counts differ (MALA heights in particular).
  - GPR-SD: P = 51.2 samples, delta_z = depth_per_sample_m_rough.
Reads metadata tables only. Writes JSON (--out), last stdout line is JSON.
"""

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FOURTU_LINES = REPO / "docs" / "processing" / "fourtu_lines_v1.csv"
GPR_DATA = Path(r"C:\workspace\GPR_Data")
GZ_INDEX = GPR_DATA / "광저우대_중국_관로GPR_가공본_261016" / "01_메타데이터" / "radargrams_index.csv"
SD_INDEX = GPR_DATA / "GPR-SD_중국USTC_도로GPR_가공본_261016" / "01_메타데이터" / "radargrams_index.csv"

TILE_PX = 256
O_MIN_PX = 64
O_MAX_PX = 224
A_MAX_M = 1.0
OVERLAP_WIDTH_PERIODS = 20.0
COMPRESS_MAX = 1.25
C_LIGHT_M_PER_NS = 0.2998
P_STARS = (4.06, 6.0)
O_CV = (None, 128, 192)  # None = physics lower bound only
FOURTU_PERIOD_NS = 2.51
GZ_PERIOD_NS = {30: 2.93, 50: 2.11}
GZ_MALA_PERIOD_NS = 2.5
GZ_VELOCITY = 0.10
SD_PERIOD_SAMPLES = 51.2


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fourtu_lines() -> list[dict]:
    out = []
    for r in read_csv(FOURTU_LINES):
        dt = float(r["dt_ns"])
        v = C_LIGHT_M_PER_NS / math.sqrt(float(r["eps_r_used"]))
        out.append({"n_traces": int(r["n_traces"]), "n_samples": int(r["n_samples"]),
                    "period_samples": FOURTU_PERIOD_NS / dt, "delta_z_m": v * dt / 2,
                    "dx_m": 0.02, "group": "all"})
    return out


def guangzhou_lines() -> list[dict]:
    out = []
    for r in read_csv(GZ_INDEX):
        if r.get("target") != "pipe" or not r.get("n_traces"):
            continue
        n_traces = int(float(r["n_traces"]))
        if n_traces <= 0:
            continue
        dt = float(r["dt_ns"])
        window = float(r["time_window_ns"]) if r.get("time_window_ns") else None
        freq = float(r["antenna_MHz"]) if r.get("antenna_MHz") else None
        if freq is None:
            period_ns = GZ_MALA_PERIOD_NS
        elif freq == 400 and window is not None and round(window) in GZ_PERIOD_NS:
            period_ns = GZ_PERIOD_NS[round(window)]
        else:
            period_ns = 1000.0 / freq
        out.append({"n_traces": n_traces, "n_samples": int(float(r["n_samples"])),
                    "period_samples": period_ns / dt, "delta_z_m": GZ_VELOCITY * dt / 2,
                    "dx_m": float(r["trace_dx_m"]),
                    "group": f"{int(freq)}MHz" if freq else "MALA"})
    return out


def gprsd_lines() -> list[dict]:
    return [{"n_traces": int(float(r["n_traces"])), "n_samples": int(float(r["n_samples"])),
             "period_samples": SD_PERIOD_SAMPLES,
             "delta_z_m": float(r["depth_per_sample_m_rough"]),
             "dx_m": float(r["trace_dx_m"]), "group": "all"}
            for r in read_csv(SD_INDEX)]


def line_tiles(line: dict, p_star: float, o_cv: int | None) -> dict:
    s_y = line["period_samples"] / p_star
    s_x = s_y * line["delta_z_m"] / line["dx_m"]
    compression = 1.0
    if round(line["n_samples"] / s_y) > TILE_PX:
        compression = (line["n_samples"] / TILE_PX) / s_y
        s_y *= compression
        s_x *= compression
    height = max(1, round(line["n_samples"] / s_y))
    width = max(1, round(line["n_traces"] / s_x))
    sigma_x = line["n_traces"] / width
    metre_term = 2 * A_MAX_M / (sigma_x * line["dx_m"])
    period_term = OVERLAP_WIDTH_PERIODS * p_star
    o_phys = max(O_MIN_PX, math.ceil(min(metre_term, period_term) - 1e-9))
    overlap = o_phys if o_cv is None else max(o_phys, o_cv)
    overlap = min(O_MAX_PX, overlap)
    n = 1 if width <= TILE_PX else 1 + math.ceil((width - TILE_PX) / (TILE_PX - overlap))
    return {"tiles": n, "o_phys": o_phys, "overlap": overlap, "height_px": height,
            "width_px": width, "compression": compression,
            "period_capped": period_term < metre_term, "single_tile": width <= TILE_PX}


def summarize(name: str, lines: list[dict]) -> dict:
    out = {"dataset": name, "n_lines": len(lines),
           "n_traces_total": sum(line["n_traces"] for line in lines), "conditions": {}}
    for p_star in P_STARS:
        for o_cv in O_CV:
            rows = [line_tiles(line, p_star, o_cv) for line in lines]
            groups: dict[str, list[int]] = {}
            for line, r in zip(lines, rows):
                g = groups.setdefault(line["group"], [0, 0, 0])
                g[0] += 1
                g[1] += line["n_traces"]
                g[2] += r["tiles"]
            compressed = [(line, r) for line, r in zip(lines, rows) if r["compression"] > 1.0]
            o_phys = sorted(r["o_phys"] for r in rows)
            key = f"P{p_star:g}_O{'phys' if o_cv is None else o_cv}"
            out["conditions"][key] = {
                "tiles_total": sum(r["tiles"] for r in rows),
                "single_tile_lines": sum(r["single_tile"] for r in rows),
                "o_phys_px_min_med_max": [o_phys[0], o_phys[len(o_phys) // 2], o_phys[-1]],
                "overlap_used_px_max": max(r["overlap"] for r in rows),
                "period_capped_lines": sum(r["period_capped"] for r in rows),
                "period_capped_groups": sorted({line["group"] for line, r in zip(lines, rows) if r["period_capped"]}),
                "compressed_lines": len(compressed),
                "compressed_groups": sorted({line["group"] for line, _ in compressed}),
                "scale_out_of_range_lines": sum(r["compression"] > COMPRESS_MAX for _, r in compressed),
                "compression_max": round(max((r["compression"] for _, r in compressed), default=1.0), 3),
                "content_height_px_min_max": [min(r["height_px"] for r in rows), max(r["height_px"] for r in rows)],
                "n2048_height_px_min_max": [
                    min((r["height_px"] for line, r in zip(lines, rows) if line["n_samples"] == 2048), default=None),
                    max((r["height_px"] for line, r in zip(lines, rows) if line["n_samples"] == 2048), default=None)],
                "by_group_lines_traces_tiles": groups,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "spec": "docs/processing/spec_detection_image_v1.md v0.3 §4-4 3, 9, 10",
        "assumptions": {
            "fourtu_period_ns": FOURTU_PERIOD_NS, "gz_period_ns": GZ_PERIOD_NS,
            "gz_mala_period_ns": GZ_MALA_PERIOD_NS, "gz_velocity_m_per_ns": GZ_VELOCITY,
            "gprsd_period_samples": SD_PERIOD_SAMPLES, "a_max_m": A_MAX_M,
            "overlap_width_periods": OVERLAP_WIDTH_PERIODS, "o_min_px": O_MIN_PX,
            "o_max_px": O_MAX_PX, "compress_max": COMPRESS_MAX, "tile_px": TILE_PX,
        },
        "fourtu": summarize("4TU", fourtu_lines()),
        "guangzhou_pipe": summarize("Guangzhou pipe", guangzhou_lines()),
        "gprsd": summarize("GPR-SD", gprsd_lines()),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: {c: v["tiles_total"] for c, v in result[k]["conditions"].items()}
             for k in ("fourtu", "guangzhou_pipe", "gprsd")}
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
