"""M2-10 (d) stage trace of the apex-candidate generator on 4TU (already-seen data, exploration).

Re-runs the rule-based generator of ``fourtu_apex_candidate_select.py`` (imported,
unchanged) on every line of the activities that have ground truth, and keeps the
items that the generator normally throws away:

* ``peak``     - every local maximum of the detection-normalised envelope
                 (``peak_local_max``, SNR >= 2, edge rule) before acceptance;
* ``accepted`` - peaks that pass the acceptance rule (``accept``) after the
                 crest move, before ranking / NMS / the per-line cap of 6;
* ``kept``     - the candidates the generator writes (must equal
                 ``fourtu_apex_candidates_v1.csv`` auto rows; checked).

Each item gets a nominal pipe-top depth ``d = v_meta * tau / 2`` (detected zero,
meta eps_r). No ground truth is read here; matching to the truth is done in
``m210d_event_selection.py``.

Output (git-ignored, large): ``data/processing/m210d_generator_items.csv`` and
``data/processing/m210d_generator_lines.csv``.

Run (single process, BelowNormal, one core, BLAS threads 1):
    conda run -n gpr-dev python -I analysis/processing/m210d_generator_stages.py
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import fourtu_apex_candidate_select as gen  # noqa: E402

from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

BASE = gen.BASE
OUT_DIR = ROOT / "data" / "processing"
ID_COLUMNS = {"activity_id": str, "path": str, "line_id": str}

ITEM_FIELDS = [
    "activity_id",
    "path",
    "peak_trace",
    "peak_sample",
    "snr",
    "trace",
    "sample",
    "tau_ns",
    "depth_nom_m",
    "stage",
    "reason",
    "n_left",
    "n_right",
    "v_straight_qc",
]
LINE_FIELDS = [
    "activity_id",
    "path",
    "n_samples",
    "n_traces",
    "dt_ns",
    "eps_r_used",
    "t_zero_ns",
    "v_meta",
    "depth_mute_m",
    "depth_window_m",
    "n_peaks",
    "n_accepted",
    "n_kept",
    "cap_active",
    "reason",
]


def gt_activities() -> list[str]:
    """Activities with at least one depth row (any code); the selection only."""
    table = pd.read_csv(
        BASE / "01_메타데이터" / "utilities_groundtruth.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return sorted({a for a, d in zip(table["activity_id"], table["depth_m"]) if d.strip()})


def line_items(act: str, path: str) -> tuple[dict, list[dict]]:
    """Replicates ``gen.apex_candidates_for_line`` and records every stage."""
    rg = load_fourtu_radargram(BASE, act, path)
    dt, dx = rg.dt_ns, rg.dx_m
    inputs = prepare_depth_inputs(rg)
    zero = inputs.zero
    v = velocity_from_eps(rg.eps_r_meta)
    line = {
        "activity_id": act,
        "path": path,
        "n_samples": rg.n_samples,
        "n_traces": rg.n_traces,
        "dt_ns": dt,
        "eps_r_used": rg.eps_r_meta,
        "t_zero_ns": None if zero.t_zero_ns is None else round(zero.t_zero_ns, 4),
        "v_meta": round(v, 5),
        "depth_mute_m": None,
        "depth_window_m": None,
        "n_peaks": 0,
        "n_accepted": 0,
        "n_kept": 0,
        "cap_active": False,
        "reason": "",
    }
    if zero.t_zero_ns is None:
        line["reason"] = "zero_failed"
        return line, []
    t_z = zero.t_zero_ns
    E = pick_signal(inputs.branch.background_removed, "envelope")
    k_lo = math.ceil((t_z + 1e9 / rg.f_c_hz) / dt) + 2
    k_hi = E.shape[0] - 16
    line["depth_mute_m"] = round(v * (k_lo * dt - t_z) / 2, 4)
    line["depth_window_m"] = round(v * ((k_hi - 1) * dt - t_z) / 2, 4)
    sub = E[k_lo:k_hi]
    noise = float(np.median(sub))
    if not noise > 0:
        line["reason"] = "flat_section"
        return line, []
    level = uniform_filter1d(
        np.median(sub, axis=1), size=max(3, int(1e9 / rg.f_c_hz / dt))
    )
    level = np.maximum(level, 0.05 * float(level.max()))
    detect = sub / level[:, None]
    peaks = peak_local_max(
        detect,
        min_distance=8,
        threshold_abs=gen.SNR_MIN,
        exclude_border=False,
        num_peaks=gen.MAX_CANDIDATES,
    )
    edge = int(math.ceil(gen.EDGE_M / dx))
    peaks = [
        (int(k) + k_lo, int(j)) for k, j in peaks if edge <= j < rg.n_traces - edge
    ]
    line["n_peaks"] = len(peaks)
    items, accepted = [], []
    for k_int, j0 in peaks:
        k0, _ = parabolic_peak(E[:, j0], k_int)
        snr = float(detect[k_int - k_lo, j0])
        item = {
            "activity_id": act,
            "path": path,
            "peak_trace": j0,
            "peak_sample": round(k0, 2),
            "snr": round(snr, 2),
            "trace": j0,
            "sample": round(k0, 2),
            "tau_ns": round(k0 * dt - t_z, 4),
            "depth_nom_m": round(v * (k0 * dt - t_z) / 2, 4),
            "stage": "peak",
            "reason": "",
            "n_left": None,
            "n_right": None,
            "v_straight_qc": None,
        }
        j, k = gen.crest_apex(E, j0, k0, dt, dx)
        if not edge <= j < rg.n_traces - edge or k < k_lo:
            item["reason"] = "apex_at_edge_or_mute"
            items.append(item)
            continue
        item.update(
            trace=j,
            sample=round(k, 2),
            tau_ns=round(k * dt - t_z, 4),
            depth_nom_m=round(v * (k * dt - t_z) / 2, 4),
            sample_raw=k,
        )
        k_c =min(int(round(k)), k_hi - 1)
        vq, nl, nr, why = gen.accept(E, j, k, dt, dx, t_z)
        item.update(n_left=nl, n_right=nr)
        if vq is None:
            item["reason"] = why
            items.append(item)
            continue
        item.update(
            stage="accepted",
            v_straight_qc=round(vq, 4),
            snr=round(float(detect[k_c - k_lo, j]), 1),
        )
        items.append(item)
        accepted.append(item)
    line["n_accepted"] = len(accepted)
    order = sorted(accepted, key=lambda s: (-min(s["n_left"], s["n_right"]), -s["snr"]))
    kept: list[dict] = []
    for s in order:
        if all(
            abs(s["trace"] - o["trace"]) * dx > gen.NMS_M
            or abs(round(s["sample_raw"] * dt, 3) - round(o["sample_raw"] * dt, 3)) > gen.NMS_NS
            for o in kept
        ):
            kept.append(s)
        if len(kept) >= gen.MAX_APEX_CANDIDATES_PER_LINE:
            line["cap_active"] = True
            break
    for s in accepted:
        if s in kept:
            s["stage"] = "kept"
        else:
            # suppressed by NMS of a better-ranked candidate, or after the cap
            s["reason"] = "nms_or_cap"
    # cap_active: the cap stopped the ranking while unranked accepted items remained
    line["cap_active"] = bool(
        line["cap_active"] and len(kept) == gen.MAX_APEX_CANDIDATES_PER_LINE
        and len(accepted) > len(kept)
    )
    line["n_kept"] = len(kept)
    return line, items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", default="")
    args = parser.parse_args()
    index = pd.read_csv(
        BASE / "01_메타데이터" / "radargrams_index.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    acts = gt_activities()
    if args.activities:
        acts = [a for a in acts if a in set(args.activities.split(","))]
    pairs = [(a, p) for a, p in zip(index["activity_id"], index["path"]) if a in acts]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines, items = [], []
    for n, (act, path) in enumerate(pairs, 1):
        line, its = line_items(act, path)
        lines.append(line)
        items.extend(its)
        if n % 25 == 0 or n == len(pairs):
            print(f"[{n}/{len(pairs)}] {act}/{path} elapsed {time.perf_counter() - started:.0f} s", flush=True)
    suffix = "" if not args.activities else "_subset"
    with open(OUT_DIR / f"m210d_generator_items{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    with open(OUT_DIR / f"m210d_generator_lines{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINE_FIELDS)
        w.writeheader()
        w.writerows(lines)
    # reproduction check against the committed candidate csv (auto rows)
    cand = pd.read_csv(ROOT / "docs" / "processing" / "fourtu_apex_candidates_v1.csv", dtype=ID_COLUMNS)
    cand = cand[(cand["source"] == "auto_v1") & cand["activity_id"].isin(acts)]
    mine = pd.DataFrame([i for i in items if i["stage"] == "kept"])
    key = lambda df: set(zip(df["activity_id"], df["path"], df["trace"].astype(int), df["sample"].round(2)))  # noqa: E731
    a, b = key(cand), key(mine)
    print(f"kept reproduced: csv {len(a)}, rerun {len(b)}, only_csv {len(a - b)}, only_rerun {len(b - a)}")
    print(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
