"""M2-9 C follow-up: size and source of near-apex pick jitter on 4TU true apexes.

Diagnosis only (labels choose cases; no threshold is set here). For every
true apex (type T in docs/processing/m29c_apex_review.csv):

  * ridge steps: per trace within +-0.12 m of the apex candidate (6 traces per
    side), the parabolic envelope maximum within +-0.5 period of the apex
    time; step dk_j = t_{j+1} - t_j minus the step of the meta-v straight-ray
    hyperbola through the apex (removes the geometric trend).
  * zero-band steps: per-trace lag of the zero band (eval_depth_4tu.
    zero_band_track, signed cross-correlation with the median template) at
    the same traces. A trace-wise timebase (sampler / trigger) jitter moves
    both by the same amount; clutter moves only the ridge.
  * line-wide zero-band step noise: 1.4826 MAD of all adjacent lag steps.

Also, for all picks of the depth path (extract_picks), residuals to the QC
curvature fit restricted to |x - x0| <= 0.12 m and their robust sigma.

Outputs: docs/processing/m29c_track_jitter.csv (per case),
         docs/processing/m29c_track_jitter.json (pooled numbers).

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_jitter.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

from eval_depth_4tu import candidate_box, zero_band_track  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.picks import ApexHint, extract_picks  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = PROJECT / "docs" / "processing"
NEAR_M = 0.12
MAD = 1.4826


def robust_sigma(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan
    return MAD * float(np.median(np.abs(v - np.median(v))))


def main() -> None:
    config = DepthConfig()
    review = pd.read_csv(DOCS / "m29c_apex_review.csv", dtype={"activity_id": str})
    track = pd.read_csv(DOCS / "m29c_track_cases.csv", dtype={"activity_id": str})
    group = track.drop_duplicates("blind_id").set_index("blind_id")["group"]
    cases = review[review["type"] == "T"]
    rows = []
    pooled = {"ridge": [], "zero": [], "ridge_minus_zero": []}
    line_noise = {}
    for (act, path), g in cases.groupby(["activity_id", "path"], sort=True):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg, config)
        dt, dx = rg.dt_ns, rg.dx_m
        period = 1e9 / rg.f_c_hz
        zt = zero_band_track(inputs.branch.filtered, inputs.zero.t_zero_ns, dt, rg.f_c_hz)
        lag = zt[0] if zt is not None else np.full(rg.n_traces, np.nan)
        line_noise[f"{act}/{path}"] = robust_sigma(np.diff(lag))
        env = pick_signal(inputs.branch.background_removed, config.pick_mode)
        v_m = velocity_from_eps(rg.eps_r_meta)
        for row in g.itertuples():
            trace, sample = round(float(row.trace)), float(row.sample)
            box = candidate_box(rg, trace, sample, float(row.tau_ns))
            picks = extract_picks(
                inputs.branch.background_removed, box, inputs.zero, dt_ns=dt, dx_m=dx,
                f_c_hz=rg.f_c_hz, config=config, apex_hint=ApexHint(trace, sample),
                period_px=inputs.branch.period_px,
            )
            rec = {"blind_id": row.blind_id, "group": group.get(row.blind_id), "line": f"{act}/{path}"}
            rec["zero_step_sigma_line_ns"] = line_noise[f"{act}/{path}"]
            if picks.apex is None:
                rows.append(rec)
                continue
            ja, ka = picks.apex.trace, picks.apex.k
            tau_a = ka * dt - inputs.zero.t_zero_ns
            d_m = v_m * tau_a / 2
            half = round(NEAR_M / dx)
            lo = max(1, math.floor(ka - 0.5 * period / dt))
            hi = min(env.shape[0] - 2, math.ceil(ka + 0.5 * period / dt))
            tr = np.arange(max(0, ja - half), min(rg.n_traces, ja + half + 1))
            kr = []
            for j in tr:
                col = env[:, j]
                kp = lo + int(np.argmax(col[lo : hi + 1]))
                kr.append(parabolic_peak(col, kp)[0] if kp not in (lo, hi) else math.nan)
            kr = np.array(kr) * dt
            geo = 2.0 / v_m * np.hypot(d_m, (tr - ja) * dx)
            ridge_step = np.diff(kr - geo)
            zero_step = np.diff(lag[tr])
            ok = np.isfinite(ridge_step) & np.isfinite(zero_step)
            pooled["ridge"].extend(ridge_step[ok])
            pooled["zero"].extend(zero_step[ok])
            pooled["ridge_minus_zero"].extend((ridge_step - zero_step)[ok])
            rec.update(
                n_steps=int(ok.sum()),
                ridge_step_sigma_ns=robust_sigma(ridge_step[ok]),
                ridge_step_rms_ns=float(np.sqrt(np.mean(ridge_step[ok] ** 2))) if ok.any() else math.nan,
                ridge_step_min_ns=float(np.min(ridge_step[ok])) if ok.any() else math.nan,
                zero_step_sigma_ns=robust_sigma(zero_step[ok]),
                corr_ridge_zero=float(np.corrcoef(ridge_step[ok], zero_step[ok])[0, 1]) if ok.sum() > 3 else math.nan,
                ridge_minus_zero_sigma_ns=robust_sigma((ridge_step - zero_step)[ok]),
                frac_steps_below_m0p1=float(np.mean(ridge_step[ok] < -0.1)) if ok.any() else math.nan,
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(DOCS / "m29c_track_jitter.csv", index=False, float_format="%.4f")
    r = np.array(pooled["ridge"])
    z = np.array(pooled["zero"])
    rz = np.array(pooled["ridge_minus_zero"])
    summary = {
        "near_m": NEAR_M,
        "n_steps": int(r.size),
        "ridge_step_sigma_ns": robust_sigma(r),
        "ridge_step_rms_ns": float(np.sqrt(np.mean(r**2))),
        "ridge_step_q": np.quantile(r, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(4).tolist(),
        "frac_ridge_steps_below_minus_0p1ns": float(np.mean(r < -0.1)),
        "frac_ridge_steps_below_minus_0p3ns": float(np.mean(r < -0.3)),
        "zero_step_sigma_ns": robust_sigma(z),
        "corr_ridge_zero": float(np.corrcoef(r, z)[0, 1]),
        "ridge_minus_zero_sigma_ns": robust_sigma(rz),
        "line_zero_step_sigma_median_ns": float(np.nanmedian(list(line_noise.values()))),
        "line_zero_step_sigma_max_ns": float(np.nanmax(list(line_noise.values()))),
        "by_group": {
            k: {
                "ridge_step_sigma_ns_median": float(gg["ridge_step_sigma_ns"].median()),
                "frac_below_m0p1_median": float(gg["frac_steps_below_m0p1"].median()),
                "corr_median": float(gg["corr_ridge_zero"].median()),
            }
            for k, gg in out.groupby("group")
        },
    }
    (DOCS / "m29c_track_jitter.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
