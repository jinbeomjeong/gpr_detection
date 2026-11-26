"""Shared helpers for the M3-6 GPR-SD scale diagnostics (D0-D5).

Plan: docs/processing/m36_gprsd_plan.md (v0.1 + section 11 decisions, commit e5e92e9).
Units: samples (time axis) and traces (along-line axis); metres only via dx 0.03 m
(documented, unverified). No ns values are used except where dt hypotheses A/B are named.

Region rule (Q3): D1 and D2 use only traces outside [mark - 400, mark + 400] of every
pipeline mark of the road (label window +-300 plus margin +-100).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

SD = Path(r"C:\workspace\GPR_Data\GPR-SD_중국USTC_도로GPR_가공본_261016")
NPY = SD / "05_radargrams_npy"
OUT_DIR = Path(r"C:\workspace\gpr_detection\analysis\processing")
FIG_DIR = Path(r"C:\workspace\gpr_detection\docs\processing")
# Full outputs with per-diffractor trace/sample coordinates stay local (git-ignored /data/).
# GPR-SD licence is unknown (docs/datasets/cards/gprsd.md), so the public repository gets
# summary statistics only (user decision 2026-11-26).
DERIVED_DIR = Path(r"C:\workspace\gpr_detection\data\derived\m36_gprsd")
DERIVED_REL = "data/derived/m36_gprsd"
# Top-level keys withheld from the public copy, per output file.
PRIVATE_KEYS: dict[str, tuple[str, ...]] = {
    "m36_gprsd_diffractor_fit.json": ("diffractors", "picks"),
    "m36_gprsd_diffractor_fit_strict.json": ("diffractors", "picks"),
    "m36_gprsd_pilot.json": ("windows",),   # pipe-mark (label) window positions
}
# Keys that must not appear anywhere in a public copy (per-diffractor / mark-window coordinates).
COORD_KEYS = ("u0", "cand_trace", "cand_sample", "env_peak_sample", "gate", "picks", "diffractors",
              "u", "t", "windows", "xlsx_row", "trace")

SEED = 20261126
DX_M = 0.03                 # documented trace spacing (unverified)
DZ_META = 0.012             # depth_per_sample_m_rough (provider constant)
DT_A = 97.0 / 321           # ns, hypothesis A
DT_B = 97.0 / 380           # ns, hypothesis B
C_M_PER_NS = 0.299792458
LABEL_HALF = 300            # M3-4 label window half width (traces)
MARGIN = 100                # Q3 margin
EXCL_HALF = LABEL_HALF + MARGIN
BG_WINDOW = 101             # moving-median window along traces (plan D1a)
ROADS = ("AsphaltRoad1", "AsphaltRoad2", "CementRoad1", "CementRoad2")


def line_ids() -> list[str]:
    return sorted(p.stem for p in NPY.glob("*.npy"))


def road_of(fid: str) -> str:
    return fid.rsplit("_", 1)[0]


def load_line(fid: str) -> np.ndarray:
    """Stored amplitude [321, N] as float64 (read-only source, copied)."""
    return np.asarray(np.load(NPY / f"{fid}.npy", mmap_mode="r"), dtype=np.float64)


def line_meta(fid: str) -> dict:
    return json.loads((NPY / f"{fid}.json").read_text(encoding="utf-8"))


def pipe_marks() -> dict[str, list[int]]:
    """Pipeline mark traces per road (marks.csv)."""
    out: dict[str, list[int]] = {r: [] for r in ROADS}
    with open(SD / "01_메타데이터" / "marks.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["type"] == "pipeline":
                out[r["road"]].append(int(r["location_trace"]))
    return out


def excluded_mask(fid: str, n_traces: int, marks: dict[str, list[int]] | None = None) -> np.ndarray:
    """True for traces inside any pipeline mark +-EXCL_HALF (not usable for D1/D2)."""
    marks = pipe_marks() if marks is None else marks
    m = np.zeros(n_traces, dtype=bool)
    for t in marks[road_of(fid)]:
        m[max(0, t - EXCL_HALF):min(n_traces, t + EXCL_HALF + 1)] = True
    return m


def remove_background(a: np.ndarray, window: int = BG_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the moving median along traces. Returns (residual, background)."""
    bg = ndimage.median_filter(a, size=(1, window), mode="nearest")
    return a - bg, bg


def bandpass(a: np.ndarray, p_short: float, p_long: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along samples, pass band periods [p_short, p_long] samples."""
    sos = signal.butter(order, [2.0 / p_long, 2.0 / p_short], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, a, axis=0)


def mean_spectrum(a: np.ndarray, pad_factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Trace-averaged |rfft| of DC-removed traces (same estimator as measure_period_samples)."""
    x = a - a.mean(axis=0, keepdims=True)
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(pad_factor * n)))
    spec = np.abs(np.fft.rfft(x, nfft, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(nfft, d=1.0)
    return f, spec


def spectrum_peaks(f: np.ndarray, spec: np.ndarray, split: float = 30.0, longest: float = 128.0) -> dict:
    """Overall peak, short peak (period < split) and long peak (period >= split)."""
    ok = f >= 1.0 / longest
    fk, sk = f[ok], spec[ok]
    per = 1.0 / fk
    i_all = int(np.argmax(sk))
    # local maxima only (an end bin of a sub-band is not a peak)
    loc, _ = signal.find_peaks(sk)
    loc_s = loc[per[loc] < split]
    loc_l = loc[per[loc] >= split]
    i_s = int(loc_s[np.argmax(sk[loc_s])]) if loc_s.size else None
    i_l = int(loc_l[np.argmax(sk[loc_l])]) if loc_l.size else None
    p = sk ** 2
    nan = float("nan")
    return {
        "peak_period": float(per[i_all]),
        "short_peak_period": float(per[i_s]) if i_s is not None else nan,
        "short_peak_rel_amp": float(sk[i_s] / sk[i_all]) if i_s is not None else nan,
        "long_peak_period": float(per[i_l]) if i_l is not None else nan,
        "long_peak_rel_amp": float(sk[i_l] / sk[i_all]) if i_l is not None else nan,
        "short_over_long_amp": float(sk[i_s] / sk[i_l]) if (i_s is not None and i_l is not None) else nan,
        "power_share_period_lt20": float(p[per < 20].sum() / p.sum()),
        "centroid_period": float(1.0 / ((fk * p).sum() / p.sum())),
    }


def qstats(vals, digits: int = 4) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "p10": round(float(np.percentile(a, 10)), digits),
            "p25": round(float(np.percentile(a, 25)), digits),
            "p50": round(float(np.median(a)), digits),
            "p75": round(float(np.percentile(a, 75)), digits),
            "p90": round(float(np.percentile(a, 90)), digits),
            "min": round(float(a.min()), digits), "max": round(float(a.max()), digits)}


def road_bootstrap(values_by_road: dict[str, np.ndarray], stat=np.median, n_boot: int = 2000,
                   seed: int = SEED) -> dict:
    """Cluster bootstrap over roads (resample roads with replacement, pool, apply stat)."""
    rng = np.random.default_rng(seed)
    roads = [r for r in values_by_road if len(values_by_road[r])]
    pooled = np.concatenate([values_by_road[r] for r in roads])
    est = float(stat(pooled))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(roads), len(roads))
        boots[b] = stat(np.concatenate([values_by_road[roads[i]] for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"estimate": est, "ci95": [float(lo), float(hi)], "half_width": float((hi - lo) / 2),
            "half_width_rel": float((hi - lo) / 2 / abs(est)) if est != 0 else None,
            "n_roads": len(roads), "n_values": int(pooled.size), "n_boot": n_boot, "seed": seed}


def dump_json(obj: dict) -> str:
    """Serialisation used by every M3-6 output (one place, so split copies stay byte-comparable)."""
    return json.dumps(obj, ensure_ascii=False, indent=1)


def write_json(name: str, obj: dict) -> Path:
    p = OUT_DIR / name
    p.write_text(dump_json(obj), encoding="utf-8")
    return p


def public_view(name: str, obj: dict) -> dict:
    """Public copy: the full object without the withheld top-level keys, plus a pointer block.

    Retained values are the same Python objects (no rounding or re-computation).
    """
    withheld = PRIVATE_KEYS[name]
    pub = {k: v for k, v in obj.items() if k not in withheld}
    what = ("pipe-mark (label) window positions are" if name == "m36_gprsd_pilot.json"
            else "per-diffractor trace/sample coordinates are")
    pub["public_split"] = {
        "withheld_keys": [k for k in withheld if k in obj],
        "full_file_local": f"{DERIVED_REL}/{name}",
        "reason": f"GPR-SD licence unknown: {what} kept local "
                  "(git-ignored); this copy holds summary statistics only (user decision 2026-11-26)",
    }
    return pub


def write_json_split(name: str, obj: dict) -> tuple[Path, Path]:
    """Write the full object to DERIVED_DIR and the public view to OUT_DIR.

    Returns (public_path, full_path).
    """
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    full = DERIVED_DIR / name
    full.write_text(dump_json(obj), encoding="utf-8")
    pub = write_json(name, public_view(name, obj))
    return pub, full


def load_full_json(name: str) -> dict:
    """Read the local full output (with per-diffractor tables); fail clearly if absent."""
    p = DERIVED_DIR / name
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found: the full M3-6 output is local only (git-ignored). Re-run the producing "
            f"script or analysis/processing/m36_gprsd_split_public.py on a pre-split copy.")
    return json.loads(p.read_text(encoding="utf-8"))
