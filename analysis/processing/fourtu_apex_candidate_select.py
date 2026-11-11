"""Rule-based apex candidates for every 4TU line (M2-8 preparation). No ground truth is read.

Per line (spec_depth_v1 v1.7 processing, src functions, default settings):

1. ``prepare_depth_inputs`` -> background-removed (median 301) section and the
   detected zero ``t_z`` (air-coupled rule).
2. Envelope ``E`` below ``k_mute + 2`` (``t_z + 1/f_c``) and above the last 16
   samples.
3. Candidates on a detection-only normalised envelope (``E`` divided by the
   per-sample median over traces, smoothed over one period):
   ``peak_local_max(min_distance=8, threshold_abs=SNR_MIN)``, at most
   ``MAX_CANDIDATES`` strongest, at least ``EDGE_M`` from both line ends.
   Each candidate is moved along its crest (+-0.5 m, earlier moves allowed)
   to the earliest point (``crest_apex``).
4. Each candidate is tracked with the spec 5-3 rules (window
   ``[prev - 0.3, prev + 2 dx / v_min + 0.3]`` ns, stop below ``0.2 E_a``,
   ``|x - x_a| <= 1.0 m``, stop when moving > 0.1 ns earlier).
5. Accept as a hyperbola apex when: >= 3 picks on each side and >= 9 in total
   (spec 5-4); the candidate is the earliest pick within +-0.10 m (apex, not
   a wing); and a parabola ``tau = p0 + p1 xi + p2 xi^2`` over |xi| <= 0.3 m
   has ``p2 > 0`` with straight-ray ``v = sqrt(2 / (p2 tau_a))`` in
   [0.04, 0.20] m/ns (loose physical range; QC only, never a depth); a V
   (limb crossing) fit must not beat the hyperbola by ``V_SHAPE_RATIO``.
6. Ranking by two-sided support (min of wing pick counts), then SNR;
   non-maximum suppression (0.3 m, 2 ns); at most ``MAX_APEX_CANDIDATES_PER_LINE``.

The 39 hand-picked apex candidates of analysis/processing/fourtu_explore.py are appended
unchanged (``source = manual_v0``) with the nearest auto candidate noted.

Outputs (docs/processing/):
* ``fourtu_apex_candidates_v1.csv``: one row per candidate.
* ``fourtu_apex_candidate_lines_v1.csv``: one row per line (zero rule, counts, reason
  when a line has no candidate).
Figures come from analysis/processing/fourtu_apex_candidate_overview.py
(``fourtu_apex_candidates_v1_pilot.png``, ``fourtu_apex_candidates_v1_overview_a/_b.png``) and
counts from analysis/processing/fourtu_apex_candidate_summary.py.

Single process, BelowNormal recommended. Progress per activity on stdout; the
CSVs are rewritten after every activity so a stopped run keeps its results.
Run:  conda run -n gpr-dev python -I analysis/processing/fourtu_apex_candidate_select.py [--activities 01.1,04.1]
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import m210b_floor_rules as floor_rules  # noqa: E402

from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import (  # noqa: E402
    FourTUTable,
    load_fourtu_radargram,
    load_fourtu_table,
    site_of_activity,
)
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
OUT = ROOT / "docs" / "processing"

SNR_MIN = 2.0
MAX_CANDIDATES = 300
MAX_APEX_CANDIDATES_PER_LINE = 6
EDGE_M = 0.2
A_MAX_M = 1.0
V_MIN_TRACK = 0.05
BACK_NS, REVERSE_NS, STOP_REL = 0.3, 0.1, 0.2
V_ACCEPT = (0.04, 0.20)
NMS_M, NMS_NS = 0.3, 2.0
V_SHAPE_RATIO = 2.0
"""Reject a candidate as a limb crossing only if the hyperbola RMS is > 2x the V RMS."""

MANUAL_APEX_CANDIDATES: dict[tuple[str, str], list[tuple[int, int]]] = {
    ("01.1", "Path1"): [(105, 175), (160, 125), (15, 285)],
    ("01.1", "Path2"): [(70, 180), (140, 150)],
    ("01.1", "Path3"): [(190, 145), (95, 190)],
    ("01.1", "Path4"): [(200, 150), (85, 255)],
    ("01.1", "Path5"): [(225, 130), (100, 255), (185, 110)],
    ("01.1", "Path6"): [(245, 145), (95, 245)],
    ("01.1", "Path7"): [(85, 180), (280, 140), (175, 75)],
    ("01.1", "Path8"): [(85, 175), (180, 80)],
    ("01.1", "Path9"): [(90, 170)],
    ("01.1", "Path10"): [(85, 160)],
    ("01.1", "Path11"): [(105, 150), (185, 85)],
    ("01.1", "Path12"): [(70, 165)],
    ("04.1", "Path1"): [(85, 245)],
    ("04.1", "Path3"): [(80, 228)],
    ("010.14", "Path2"): [(175, 135), (470, 165), (585, 170), (330, 190)],
    ("010.14", "Path3"): [(565, 185)],
    ("010.14", "Path4"): [(440, 140), (605, 150)],
    ("011.14", "Profile_6161"): [(165, 120), (200, 105)],
    ("011.14", "Profile_6163"): [(200, 110), (380, 125)],
    ("011.14", "Profile_6164"): [(260, 115), (140, 95)],
}
"""Copied verbatim from analysis/processing/fourtu_explore.py (trace, sample)."""

FATE_KEPT = "kept"
FATE_NMS = "nms"
FATE_CAP6 = "cap6"
"""Fate of an accepted candidate (m210b 8-2): kept, suppressed by a kept candidate
(NMS), or never examined because the per-line cap was already full."""

APEX_CANDIDATE_FIELDS = [
    "activity_id",
    "site",
    "path",
    "trace",
    "sample",
    "t_ns",
    "tau_ns",
    "snr",
    "n_left",
    "n_right",
    "v_straight_qc",
    "source",
    "note",
]
LINE_FIELDS = [
    "activity_id",
    "site",
    "path",
    "n_traces",
    "dt_ns",
    "eps_r_used",
    "zero_rule",
    "t_zero_ns",
    "n_peaks",
    "n_apex_candidates",
    "reason",
]


def crest_apex(
    E: np.ndarray, j0: int, k0: float, dt: float, dx: float
) -> tuple[int, float]:
    """Walk the crest +-0.5 m from a candidate allowing earlier moves; return its earliest point.

    An envelope maximum need not sit on the apex (amplitude can peak on a wing);
    the spec tracker stops when moving earlier, so the candidate is first moved here.
    """
    e0 = float(E[int(round(k0)), j0])
    best = (j0, k0)
    for step in (-1, 1):
        prev = k0
        for dist in range(1, int(0.5 / dx + 1e-9) + 1):
            j = j0 + step * dist
            if not 0 <= j < E.shape[1]:
                break
            lo = max(0, math.floor(prev - 1.0 / dt))
            hi = min(E.shape[0] - 1, math.ceil(prev + 1.0 / dt))
            col = E[:, j]
            k, a = parabolic_peak(col, lo + int(np.argmax(col[lo : hi + 1])))
            if a < STOP_REL * e0:
                break
            if k < best[1]:
                best = (j, k)
            prev = k
    return best


def track(
    E: np.ndarray, j0: int, k0: float, step: int, dt: float, dx: float, e0: float
) -> list:
    out, prev = [], k0
    back = BACK_NS / dt
    fwd = (2 * dx / V_MIN_TRACK + BACK_NS) / dt
    for dist in range(1, int(A_MAX_M / dx + 1e-9) + 1):
        j = j0 + step * dist
        if not 0 <= j < E.shape[1]:
            break
        lo = max(0, math.floor(prev - back))
        hi = min(E.shape[0] - 1, math.ceil(prev + fwd))
        if hi <= lo:
            break
        col = E[:, j]
        k, a = parabolic_peak(col, lo + int(np.argmax(col[lo : hi + 1])))
        if a < STOP_REL * e0 or k < prev - REVERSE_NS / dt:
            break
        out.append((j, k))
        prev = k
    return out


def accept(E, j, k, dt, dx, t_z):
    e0 = float(E[int(round(k)), j])
    left = track(E, j, k, -1, dt, dx, e0)
    right = track(E, j, k, +1, dt, dx, e0)
    nl, nr = len(left), len(right)
    if nl < 3 or nr < 3 or nl + nr + 1 < 9:
        return None, nl, nr, "few_picks"
    pts = [(j, k), *left, *right]
    xi = np.array([(p[0] - j) * dx for p in pts])
    tau = np.array([p[1] * dt - t_z for p in pts])
    tau_a = k * dt - t_z
    near = np.abs(xi) <= 0.10 + 1e-9
    if tau[near].min() < tau_a - 0.05:
        return None, nl, nr, "not_apex"
    sel = np.abs(xi) <= 0.30 + 1e-9
    if sel.sum() < 5 or tau_a <= 0:
        return None, nl, nr, "short"
    p2 = np.polyfit(xi[sel], tau[sel], 2)[0]
    if p2 <= 0:
        return None, nl, nr, "not_convex"
    v = math.sqrt(2.0 / (p2 * tau_a))
    if not V_ACCEPT[0] <= v <= V_ACCEPT[1]:
        return None, nl, nr, "v_out_of_range"
    # Two crossing limbs of different hyperbolas also make a local "apex", but a
    # V (|xi|) shape, while a true apex is smooth (quadratic). Compare both.
    wide = np.abs(xi) <= 0.40 + 1e-9
    if wide.sum() >= 7:
        x_w, t_w = xi[wide], tau[wide]
        # Hyperbola with free x0: tau^2 = a + b xi + c xi^2 (straight ray, linear in a, b, c).
        sq = np.polyfit(x_w, t_w**2, 2)
        quad = np.sqrt(np.maximum(np.polyval(sq, x_w), 0.0)) - t_w
        design = np.column_stack([np.ones_like(x_w), np.abs(x_w), x_w])
        coef, *_ = np.linalg.lstsq(design, t_w, rcond=None)
        vee = design @ coef - t_w
        # Only a clearly better V rejects: 4TU apexes are shallow and sharp, so a
        # plain "V fits better" test also rejected 14 of the 39 hand-picked apex candidates.
        if np.sqrt(np.mean(quad**2)) > V_SHAPE_RATIO * np.sqrt(np.mean(vee**2)):
            return None, nl, nr, "v_shape_crossing"
    return v, nl, nr, ""


def _site_label(act: str) -> int | None:
    """4TU site of ``act``; ``None`` for a non-4TU line such as ``S1G`` (S1g)."""
    site_text = act.split(".", 1)[0]
    return site_of_activity(act) if site_text.isdigit() else None


def apex_candidates_for_line(act: str, path: str):
    """Apex candidates of one 4TU line (reads the radargram, then the wrapper)."""
    rg = load_fourtu_radargram(BASE, act, path)
    inputs = prepare_depth_inputs(rg)
    return apex_candidates_for_radargram(rg, inputs, act=act, path=path)


def _rank_and_suppress(
    accepted: list[dict], dx: float
) -> tuple[list[dict], list[dict]]:
    """Kept candidates (NMS, per-line cap) and the fate of every accepted one.

    ``accepted`` must already be in rank order. A fate is ``{"fate": FATE_*,
    "winner": index into kept or None}``; ``winner`` is the first kept candidate
    that suppressed an ``FATE_NMS`` entry.
    """
    kept: list[dict] = []
    fates: list[dict] = []
    for s in accepted:
        if len(kept) >= MAX_APEX_CANDIDATES_PER_LINE:
            fates.append({"fate": FATE_CAP6, "winner": None})
            continue
        winner = next(
            (
                index
                for index, o in enumerate(kept)
                if abs(s["trace"] - o["trace"]) * dx <= NMS_M
                and abs(s["t_ns"] - o["t_ns"]) <= NMS_NS
            ),
            None,
        )
        if winner is None:
            kept.append(s)
            fates.append({"fate": FATE_KEPT, "winner": len(kept) - 1})
        else:
            fates.append({"fate": FATE_NMS, "winner": winner})
    return kept, fates


def apex_candidates_for_radargram(
    rg,
    inputs,
    *,
    act: str,
    path: str,
    floor_rule: str = "current",
    mute_offset_ns: float = 0.0,
):
    """Apex candidates of a loaded radargram and its depth inputs (spec_eval 9-9-1 (b)).

    The generator body of ``apex_candidates_for_line`` after loading, so a
    synthetic B-scan (S1g, S4/S6) goes through the same rules as a 4TU line. The
    defaults reproduce selector v1 exactly (m210b 8-2).

    Args:
        rg: The radargram.
        inputs: ``prepare_depth_inputs(rg)``.
        act: Activity label of the line rows.
        path: Path label of the line rows.
        floor_rule: Detection-normalisation floor rule of
            ``m210b_floor_rules.RULES`` (``"current"`` = selector v1).
        mute_offset_ns: Shift of the mute only: the search starts at
            ``t_z + mute_offset_ns + 1/f_c``. ``tau_ns`` and the acceptance keep
            the detected zero ``t_z``.

    Returns:
        ``(line, kept, E)``: the line row, the kept candidate rows and the envelope
        (``None`` when the zero failed). When candidates were searched, ``line``
        also carries ``_accepted_all`` (every accepted candidate, rank order),
        ``_accepted_fate`` (parallel list, see ``_rank_and_suppress``), ``_peaks``
        (every detection peak after the edge rule: its ``row`` and ``trace0``,
        the refined start ``sample0``, the crest-moved ``trace``/``sample`` and
        ``status`` = ``"accepted"`` or the reject reason) and ``_floor`` (search
        rows, ``L``, ``F``, the unfloored level profile and the raw peak count
        before the edge rule).
    """
    dt, dx = rg.dt_ns, rg.dx_m
    zero = inputs.zero
    line = {
        "activity_id": act,
        "site": _site_label(act),
        "path": path,
        "n_traces": rg.n_traces,
        "dt_ns": dt,
        "eps_r_used": rg.eps_r_meta,
        "zero_rule": zero.zero_rule,
        "t_zero_ns": None if zero.t_zero_ns is None else round(zero.t_zero_ns, 3),
        "n_peaks": 0,
        "n_apex_candidates": 0,
        "reason": "",
    }
    if zero.t_zero_ns is None:
        line["reason"] = "zero_failed"
        return line, [], None
    background_removed = inputs.branch.background_removed
    E = pick_signal(background_removed, "envelope")
    k_lo, k_hi = floor_rules.search_rows(
        zero.t_zero_ns + mute_offset_ns, rg.f_c_hz, dt, E.shape[0]
    )
    sub = E[k_lo:k_hi]
    noise = float(np.median(sub))
    if not noise > 0:
        line["reason"] = "flat_section"
        return line, [], E
    # Detection-only normalisation (a gain): divide by the per-sample median over
    # traces, smoothed over one period, so deep weak apexes compete with shallow
    # clutter. Pick times always come from the un-normalised envelope E.
    level = uniform_filter1d(
        np.median(sub, axis=1), size=max(3, int(1e9 / rg.f_c_hz / dt))
    )
    floor = floor_rules.floor_value(
        floor_rule,
        level,
        sub,
        background_removed[k_lo:k_hi],
        rg.f_c_hz,
        dt,
        background_removed[:k_hi],
    )
    line["_floor"] = {
        "k_lo": k_lo,
        "k_hi": k_hi,
        "L": float(level.max()),
        "F": float(floor),
        "level": level,
    }
    level = np.maximum(level, floor)
    detect = sub / level[:, None]
    peaks = peak_local_max(
        detect,
        min_distance=8,
        threshold_abs=SNR_MIN,
        exclude_border=False,
        num_peaks=MAX_CANDIDATES,
    )
    line["_floor"]["n_raw_peaks"] = len(peaks)
    edge = int(math.ceil(EDGE_M / dx))
    peaks = [
        (int(k) + k_lo, int(j)) for k, j in peaks if edge <= j < rg.n_traces - edge
    ]
    line["n_peaks"] = len(peaks)
    accepted, reasons, peak_records = [], {}, []
    for k_int, j0 in peaks:
        k0, _ = parabolic_peak(E[:, j0], k_int)
        j, k = crest_apex(E, j0, k0, dt, dx)
        peak = {"row": k_int, "trace0": j0, "sample0": k0, "trace": j, "sample": k}
        peak_records.append(peak)
        if not edge <= j < rg.n_traces - edge or k < k_lo:
            reasons["apex_at_edge_or_mute"] = reasons.get("apex_at_edge_or_mute", 0) + 1
            peak["status"] = "apex_at_edge_or_mute"
            continue
        k_int = min(int(round(k)), k_hi - 1)
        v, nl, nr, why = accept(E, j, k, dt, dx, zero.t_zero_ns)
        peak["status"] = why or "accepted"
        if v is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        accepted.append(
            {
                "activity_id": act,
                "site": _site_label(act),
                "path": path,
                "trace": j,
                "sample": round(k, 2),
                "t_ns": round(k * dt, 3),
                "tau_ns": round(k * dt - zero.t_zero_ns, 3),
                "snr": round(float(detect[k_int - k_lo, j]), 1),
                "n_left": nl,
                "n_right": nr,
                "v_straight_qc": round(v, 4),
                "source": "auto_v1",
                "note": "",
            }
        )
    # Rank by two-sided support (both wings tracked far), then by detection SNR:
    # short limb crossings rarely have long wings on both sides.
    accepted.sort(key=lambda s: (-min(s["n_left"], s["n_right"]), -s["snr"]))
    kept, fates = _rank_and_suppress(accepted, dx)
    line["n_apex_candidates"] = len(kept)
    if not kept:
        line["reason"] = (
            "no_candidate"
            if not peaks
            else "rejected:" + ";".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        )
    line["_accepted_all"] = accepted
    line["_accepted_fate"] = fates
    line["_peaks"] = peak_records
    line["_diag"] = (E, dt, dx, zero.t_zero_ns)
    return line, kept, E


def manual_rows(act, path, auto, dt, t_z, line=None):
    rows = []
    for j, k in MANUAL_APEX_CANDIDATES.get((act, path), []):
        near = [
            a for a in auto if abs(a["trace"] - j) <= 12 and abs(a["sample"] - k) <= 15
        ]
        if near:
            note = "auto match"
        else:
            note = "no auto apex candidate within 12 traces / 15 samples"
            pre = [
                a
                for a in (line or {}).get("_accepted_all", [])
                if abs(a["trace"] - j) <= 12 and abs(a["sample"] - k) <= 15
            ]
            if pre:
                note += "; accepted but dropped by NMS/per-line cap"
            elif line and "_diag" in line:
                E, dt_, dx_, tz = line["_diag"]
                # Same refinement as explore: envelope max within +-12 traces / +-15 samples.
                j0, j1 = max(0, j - 12), min(E.shape[1] - 1, j + 12)
                k0, k1 = max(0, k - 15), min(E.shape[0] - 1, k + 15)
                kk, jj = np.unravel_index(
                    np.argmax(E[k0 : k1 + 1, j0 : j1 + 1]), (k1 - k0 + 1, j1 - j0 + 1)
                )
                ja, ka = crest_apex(E, j0 + int(jj), float(k0 + kk), dt_, dx_)
                v, nl, nr, why = accept(E, ja, ka, dt_, dx_, tz)
                note += f"; rule on this apex candidate: {'accept' if v else why} (left {nl}, right {nr})"
        rows.append(
            {
                "activity_id": act,
                "site": site_of_activity(act),
                "path": path,
                "trace": j,
                "sample": k,
                "t_ns": round(k * dt, 3),
                "tau_ns": None if t_z is None else round(k * dt - t_z, 3),
                "snr": None,
                "n_left": None,
                "n_right": None,
                "v_straight_qc": None,
                "source": "manual_v0",
                "note": note,
            }
        )
    return rows


def write(rows, fields, name):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", default="")
    args = parser.parse_args()
    index = load_fourtu_table(BASE, FourTUTable.RADARGRAMS_INDEX)
    pairs = list(zip(index["activity_id"], index["path"], strict=True))
    if args.activities:
        wanted = set(args.activities.split(","))
        pairs = [p for p in pairs if p[0] in wanted]
    acts = list(dict.fromkeys(a for a, _ in pairs))
    candidates, lines = [], []
    started = time.perf_counter()
    for n, act in enumerate(acts, 1):
        for a, path in pairs:
            if a != act:
                continue
            try:
                line, kept, _ = apex_candidates_for_line(act, path)
            except Exception as exc:  # report and continue; analysis script
                line = {
                    "activity_id": act,
                    "site": site_of_activity(act),
                    "path": path,
                    "reason": f"error:{type(exc).__name__}:{exc}",
                }
                kept = []
            lines.append(line)
            candidates.extend(kept)
            if (act, path) in MANUAL_APEX_CANDIDATES:
                candidates.extend(
                    manual_rows(
                        act,
                        path,
                        kept,
                        line.get("dt_ns") or 0.0,
                        line.get("t_zero_ns"),
                        line,
                    )
                )
            for private_key in [key for key in line if key.startswith("_")]:
                line.pop(private_key)
        write(candidates, APEX_CANDIDATE_FIELDS, "fourtu_apex_candidates_v1.csv")
        write(lines, LINE_FIELDS, "fourtu_apex_candidate_lines_v1.csv")
        n_auto = sum(
            1
            for s in candidates
            if s["activity_id"] == act and s["source"] == "auto_v1"
        )
        print(
            f"[{n}/{len(acts)}] {act}: lines {sum(1 for l in lines if l['activity_id'] == act)}, "
            f"auto candidates {n_auto}, elapsed {time.perf_counter() - started:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
