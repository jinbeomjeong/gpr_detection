"""M2-10 (b) model-selection estimate of Lambda = L / A0 from gprMax pilot runs (no judgment).

Used to pick the added batch-1 models (docs/processing/m210b_selector_floor.md 4-2-7).
Not a verdict: the batch-1 M0 value is recomputed from the full runs by the harness.

* L (selector level maximum) needs a LINE: the background removal and the per-row
  median are lateral statistics, so it is computed on a no-pipe paving line of
  >= 51 traces at dx 0.02 m (>= 5 joint periods; 161 recommended = batch-1 width)
  with the selector formula of ``m210b_floor_rules`` (resampled to 0.098 ns, the
  zero from ``prepare_depth_inputs``). A short time window is fine when the L row
  is at least w/2 + 16 samples above the record end (checked: ``window_ok``).
* A0 (apex envelope) comes from ONE pipe trace at the apex x minus ONE no-pipe trace
  at the same x (same joints, same sigma), both through dewow + band-pass
  (``prepare_fitting_branch(...).filtered``; background removal on one trace would
  cancel the trace), envelope by ``pick_signal``, max within the expected time
  +-0.5/f_c. In the batch the background median over 161 traces plays the role of
  the no-pipe trace at the apex time.

Input arrays: gprMax merged h5, ``rxs/rx1/Ez`` (Iterations, ntraces), attr ``dt`` (s);
sample k at k*dt (export convention time_of_sample0_ns 0).

Run (gpr-dev, one core):
    conda run -n gpr-dev python -I analysis/processing/m210b_lambda_pilot.py \
        --line-h5 <no-pipe line h5> --pipe-h5 <pipe pilot h5> --pipe-trace 1 \
        --nopipe-h5 <no-pipe pilot h5> --nopipe-trace 1 --expected-apex-ns 32.82 \
        --out <json>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import m210b_floor_rules as fr  # noqa: E402

from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.radargram import Radargram  # noqa: E402
from gpr_detection.preprocess.filters import prepare_fitting_branch  # noqa: E402
from gpr_detection.preprocess.signals import pick_signal  # noqa: E402

DT_H = 0.098  # harness resampling (doc 4-2 rule 2)
F_C = 500e6
DELTAS = (-1.0, 0.0, 1.0, 2.35)
MIN_LINE_TRACES = 51


def read_ez(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        ez = np.asarray(f["rxs/rx1/Ez"][()], dtype=np.float64)
        dt_ns = float(f.attrs["dt"]) * 1e9
    if ez.ndim == 1:
        ez = ez[:, None]
    return ez, dt_ns


def resample(ez: np.ndarray, dt_ns: float, dt_new: float = DT_H) -> np.ndarray:
    t_old = np.arange(ez.shape[0]) * dt_ns
    t_new = np.arange(int(math.floor(t_old[-1] / dt_new)) + 1) * dt_new
    return np.column_stack([np.interp(t_new, t_old, ez[:, j]) for j in range(ez.shape[1])])


def radargram(data: np.ndarray, dx: float, air_gap_m: float, eps: float, line_id: str) -> Radargram:
    return Radargram(
        data=data.astype(np.float32), dt_ns=DT_H, dx_m=dx, air_gap="air", amplitude_processed=False,
        dataset="synthetic", line_id=line_id, f_c_hz=F_C, eps_r_meta=eps,
        eps_r_meta_source="synthetic_truth", antenna_offset_m=0.1, antenna_height_m=air_gap_m,
    )


def line_level(path: Path, dx: float, air_gap_m: float, eps: float) -> dict:
    ez, dt = read_ez(path)
    data = resample(ez, dt)
    rg = radargram(data, dx, air_gap_m, eps, "S4S6/pilot_line")
    inputs = prepare_depth_inputs(rg)
    t_z = inputs.zero.t_zero_ns
    out = {"n_traces": int(data.shape[1]), "n_samples": int(data.shape[0]), "t_zero_ns": t_z,
           "zero_rule": str(inputs.zero.zero_rule), "line_traces_ok": data.shape[1] >= MIN_LINE_TRACES}
    if t_z is None:
        out["error"] = "zero_failed"
        return out
    E = pick_signal(inputs.branch.background_removed, "envelope")
    w = fr.period_rows(F_C, DT_H)
    for d in DELTAS:
        k_lo, k_hi = fr.search_rows(t_z + d, F_C, DT_H, data.shape[0])
        level = fr.level_profile(E[k_lo:k_hi], F_C, DT_H)
        k_arg = int(np.argmax(level)) + k_lo
        out[f"delta_{d:+.2f}"] = {
            "k_lo": k_lo, "L": float(level.max()), "t_argmax_ns": k_arg * DT_H,
            "periods_after_mute": (k_arg - k_lo) * DT_H * F_C * 1e-9,
            "window_ok": bool(k_arg + w // 2 + fr.RECORD_END_TRIM < data.shape[0] - fr.RECORD_END_TRIM),
        }
    return out


def apex_a0(pipe: Path, pipe_trace: int, nopipe: Path | None, nopipe_trace: int,
            expected_ns: float, dx: float, air_gap_m: float, eps: float) -> dict:
    ez, dt = read_ez(pipe)
    tr = resample(ez[:, [pipe_trace]], dt)
    f_p = prepare_fitting_branch(radargram(tr, dx, air_gap_m, eps, "S4S6/pilot_pipe")).filtered
    if nopipe is not None:
        ez0, dt0 = read_ez(nopipe)
        tr0 = resample(ez0[:, [nopipe_trace]], dt0)
        n = min(tr.shape[0], tr0.shape[0])
        f_0 = prepare_fitting_branch(radargram(tr0[:n], dx, air_gap_m, eps, "S4S6/pilot_nopipe")).filtered
        sig = np.asarray(f_p[:n]) - np.asarray(f_0)
    else:
        sig = np.asarray(f_p)
    env = pick_signal(sig, "envelope")[:, 0]
    half = 0.5e9 / F_C
    k0 = int(math.floor((expected_ns - half) / DT_H))
    k1 = int(math.ceil((expected_ns + half) / DT_H))
    k = k0 + int(np.argmax(env[k0: k1 + 1]))
    return {"A0": float(env[k]), "t_apex_ns": k * DT_H, "subtracted_nopipe": nopipe is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-h5", type=Path)
    ap.add_argument("--pipe-h5", type=Path, required=True)
    ap.add_argument("--pipe-trace", type=int, required=True)
    ap.add_argument("--nopipe-h5", type=Path)
    ap.add_argument("--nopipe-trace", type=int, default=0)
    ap.add_argument("--expected-apex-ns", type=float, required=True)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--air-gap", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=9.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = {"note": "model-selection estimate (doc 4-2-7), not a batch-1 result",
           "apex": apex_a0(a.pipe_h5, a.pipe_trace, a.nopipe_h5, a.nopipe_trace,
                           a.expected_apex_ns, a.dx, a.air_gap, a.eps)}
    if a.line_h5 is not None:
        res["line"] = line_level(a.line_h5, a.dx, a.air_gap, a.eps)
        a0 = res["apex"]["A0"]
        res["Lambda_est"] = {k: v["L"] / a0 for k, v in res["line"].items() if k.startswith("delta_")}
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
