"""M2-9 C follow-up: synthetic reproduction of one-sided tracking failures (no 4TU data).

Analytic B-scans on the 4TU grid (500 MHz Ricker, dt 0.098 ns, dx 0.02 m,
eps_r 9, ground-coupled direct wave +3 at t_src + 0.7 ns so the zero and the
mute behave as in spec 11-1). The pipe is a point reflector (R = 0) at depth
d whose amplitude falls as (d / r)^4 along the wing (beam / spreading
**assumption**, makes the 0.2 E(t_a) stop end the wings at |x| ~ 0.5-0.7 m).
Rules are the variants of m29c_track_proto.py (imported: the same in-process
replacement, project code unchanged).

Part D (deterministic, fixed shapes; candidates for spec unit tests):
  D1 near-apex jitter: the pipe event of trace j is shifted by s_j
     (apparent envelope-peak shift of an unresolved interferer). Pattern
     s_j = +s0 for even |j - j_a|, -s0 for odd, |j - j_a| <= 6, 0 beyond.
     s0 = 0.06 ns (interferer a = s0 / (2 kappa T) ~ 0.12 < r_stop) and
     s0 = 0.12 ns (a ~ 0.23 > r_stop, the tolerance must stop it).
  D2 hint-window top truncation: D1 (s0 0.06) with the hint at the true apex
     trace and the true apex sample + 0.5 P - 0.25 dt late (true peaks of the
     odd traces sit just above the hint-window top).
  D3 crossing (negative control, no shifts): a second, equal point reflector
     0.35 m right and 0.06 m shallower; its left wing rises through the
     main right wing.
  D4 rising layer (negative control): a planar reflector, amplitude 1, whose
     time falls 0.15 ns per trace to the right, crossing the main right wing
     at x_a + 0.2 m.
  Output per case and variant: n_left, n_right, enough-for-curvature, QC r,
  apex trace error, number of right picks closer to the intruding event.

Part M (Monte Carlo, physical clutter): N_s random point scatterers per
  realization (density 40 per m of line over depth 0.1 .. d + 0.6 m),
  Gaussian amplitudes scaled so that the clutter RMS in the box region is
  c x the pipe apex amplitude, c in {0, 0.05, 0.10, 0.15, 0.20}; d in
  {0.3, 0.6} m; hint = true apex (sample + 0) or late by 0.45 P. Seeds
  0..N_SEEDS-1 (numpy default_rng(seed)); white noise SNR 40 dB.
  Output: failure rate (insufficient picks), QC verdict rates, r quantiles,
  apex trace error, per condition and variant.

Outputs: docs/processing/m29c_track_repro_det.csv, m29c_track_repro_mc.csv,
         m29c_track_repro_mc_summary.csv, m29c_track_repro.png

Run: conda run -n gpr-dev python -I analysis/processing/m29c_track_repro.py

Code version: needs the v1.8 package like m29c_track_proto.py (it stops on
the v1.9 code). The deterministic cases D0-D4 are CI tests on the v1.9 code
in tests/depth/test_tracking_v19.py (spec_depth 11-3).
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "analysis" / "processing"))
sys.path.insert(0, str(PROJECT / "tests"))

import m29c_track_proto as proto  # noqa: E402  (installs the in-process variants)
from synthetic_bscan import ricker  # noqa: E402

from gpr_detection.depth.hyperbola import fit_curvature, fixed_velocity_apex  # noqa: E402
from gpr_detection.depth.picks import ApexHint  # noqa: E402
from gpr_detection.io.radargram import Radargram, velocity_from_eps  # noqa: E402

DOCS = PROJECT / "docs" / "processing"
F_C = 500e6
P = 1e9 / F_C
DT = 0.098
DX = 0.02
EPS = 9.0
V = velocity_from_eps(EPS)
N_TR = 161
N_S = 420
T_SRC = 3.0
J_A = N_TR // 2
VARIANTS = ("v18", "med3", "descent", "descent+med3", "step")
N_SEEDS = 40
CLUTTER_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
DEPTHS = (0.3, 0.6)
HINT_LATE = (0.0, 0.45)


def place(data, times, amp):
    t = np.arange(N_S)[:, None] * DT
    data += np.asarray(amp)[None, :] * ricker(t - times[None, :], F_C)


def base_section(d, shifts=None, extra=(), clutter=None, noise=None):
    x = (np.arange(N_TR) - J_A) * DX
    data = np.zeros((N_S, N_TR))
    place(data, np.full(N_TR, T_SRC + 0.7), np.full(N_TR, 3.0))
    r = np.hypot(d, x)
    tau = 2.0 / V * r
    if shifts is not None:
        tau = tau + shifts
    place(data, T_SRC + tau, (d / r) ** 4)
    for times, amp in extra:
        place(data, T_SRC + times, amp)
    if clutter is not None:
        data += clutter
    if noise is not None:
        data += noise
    return data, tau


def radargram(data):
    return Radargram(
        data=data.astype(np.float32), dt_ns=DT, dx_m=DX, air_gap="ground",
        amplitude_processed=False, dataset="unit", line_id="m29c/track",
        f_c_hz=F_C, eps_r_meta=EPS, eps_r_meta_source="synthetic_truth", antenna_offset_m=0.1,
    )


def run_variant(name, rg, inputs, hint, true_tau_apex):
    box = proto.e4.candidate_box(rg, hint.trace, hint.sample, hint.sample * DT - inputs.zero.t_zero_ns)
    with proto.variant(name):
        picks = proto.picks_mod.extract_picks(
            inputs.branch.background_removed, box, inputs.zero, dt_ns=DT, dx_m=DX,
            f_c_hz=F_C, config=proto.CFG, apex_hint=hint, period_px=inputs.branch.period_px,
        )
    rec = {"n_left": picks.n_left, "n_right": picks.n_right, "n_picks": picks.n_picks}
    rec["enough"] = picks.is_enough_for_curvature(proto.CFG)
    if picks.apex is None:
        rec["verdict"] = "unknown"
        rec["reason"] = "no_apex"
        return rec, picks
    rec["apex_dtr"] = picks.apex.trace - J_A
    curv = fit_curvature(picks, radius_m=0.0, air_gap="ground", v_fallback=V, config=proto.CFG)
    qc = proto.e4.classify_curvature(curv, picks, V)
    rec["verdict"], rec["reason"], rec["r"] = str(qc.verdict), qc.reason, qc.r
    vertex = fixed_velocity_apex(picks, proto.CFG)
    rec["t0_fixv_minus_true_ns"] = vertex.t0_ns - (true_tau_apex - (inputs.zero.t_zero_ns - T_SRC))
    return rec, picks


def true_apex_sample(rg_inputs, tau_apex):
    """Envelope peak sample of the processed apex trace (true apex, no shift)."""
    env = proto.picks_mod.pick_signal(rg_inputs.branch.background_removed[:, [J_A]], "envelope")[:, 0]
    k_c = (T_SRC + tau_apex) / DT
    lo, hi = int(k_c - 0.5 * P / DT), int(k_c + 0.5 * P / DT)
    k = lo + int(np.argmax(env[lo : hi + 1]))
    return float(proto.parabolic_peak(env, k)[0])


def deterministic():
    rows = []
    d = 0.5
    x = (np.arange(N_TR) - J_A) * DX
    m = np.abs(np.arange(N_TR) - J_A)
    cases = {}
    for s0 in (0.06, 0.12):
        shifts = np.where(m <= 6, np.where(m % 2 == 0, s0, -s0), 0.0)
        cases[f"D1_s{s0:.2f}"] = dict(shifts=shifts, hint_late=0.0)
    # D2: shallow pipe (steep near-apex ridge), hint 3 traces right of the apex
    # and late so that the hint-window top hides the peaks of the apex and its
    # first two neighbours (window top 0.5 sample above the apex peak of trace
    # j_a + 2); no shifts. The first visible trace is j_a + 3.
    cases["D2_trunc_offset3"] = dict(shifts=None, hint_late="trunc", d=0.25, hint_dtr=3)
    tau2 = 2.0 / V * np.hypot(d - 0.06, x - 0.35)
    cases["D3_crossing"] = dict(shifts=None, hint_late=0.0, extra=[(tau2, ((d - 0.06) / np.hypot(d - 0.06, x - 0.35)) ** 4)])
    tau_main = 2.0 / V * np.hypot(d, x)
    j_c = J_A + 6
    layer = tau_main[j_c] - 0.15 * (np.arange(N_TR) - j_c)
    layer_amp = np.where(np.arange(N_TR) >= j_c - 2, 1.5, 0.0)
    cases["D4_rising_event"] = dict(shifts=None, hint_late=0.0, extra=[(layer, layer_amp)])
    cases["D0_clean"] = dict(shifts=None, hint_late=0.0)
    for cname, spec in cases.items():
        dd = spec.get("d", d)
        data, tau = base_section(dd, spec.get("shifts"), spec.get("extra", ()))
        rg = radargram(data)
        inputs = proto.est_mod.prepare_depth_inputs(rg, proto.CFG)
        k_true = true_apex_sample(inputs, 2.0 / V * dd)
        hint_tr = J_A + spec.get("hint_dtr", 0)
        if spec["hint_late"] == "trunc":
            k_nb2 = (T_SRC + tau[J_A + 2]) / DT
            k0 = math.floor(k_nb2 + 0.5)
            hint_k = k0 + 0.5 * P / DT - 0.5
        else:
            hint_k = k_true + spec["hint_late"] * P / DT
        hint = ApexHint(trace=hint_tr, sample=hint_k)
        for name in VARIANTS:
            rec, picks = run_variant(name, rg, inputs, hint, 2.0 / V * dd)
            rec.update(case=cname, variant=name, d_m=dd)
            if picks.apex is not None:
                rec["right_k_ns"] = " ".join(f"{k * DT:.2f}" for k in picks.k_pick[picks.trace_idx > picks.apex.trace][:12])
            if cname in ("D3_crossing", "D4_rising_event") and picks.apex is not None:
                intr = tau2 if cname == "D3_crossing" else layer
                tz = inputs.zero.t_zero_ns
                right = picks.trace_idx > picks.apex.trace
                t_pick = picks.k_pick[right] * DT
                main_t = T_SRC + tau_main[picks.trace_idx[right]]
                intr_t = T_SRC + intr[picks.trace_idx[right]]
                rec["right_on_intruder"] = int(np.sum(np.abs(t_pick - intr_t) < np.abs(t_pick - main_t) - 0.05))
                rec["tz"] = tz
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "m29c_track_repro_det.csv", index=False, float_format="%.4f")
    return df


def clutter_field(rng, d, c_level):
    """Random point scatterers; scaled to RMS c x apex amplitude in the box region."""
    x = (np.arange(N_TR) - J_A) * DX
    line = N_TR * DX
    n = int(round(40 * line))
    xs = rng.uniform(x[0], x[-1], n)
    zs = rng.uniform(0.1, d + 0.6, n)
    amps = rng.normal(0.0, 1.0, n)
    field = np.zeros((N_S, N_TR))
    for xi, zi, ai in zip(xs, zs, amps, strict=True):
        r = np.hypot(zi, x - xi)
        place(field, T_SRC + 2.0 / V * r, ai * (zi / r) ** 4)
    t = np.arange(N_S) * DT
    t_a = T_SRC + 2.0 / V * d
    rows = (t >= t_a - 3) & (t <= t_a + 10)
    cols = np.abs(x) <= 0.8
    rms = float(np.sqrt(np.mean(field[np.ix_(rows, cols)] ** 2)))
    return field * (c_level / rms if rms > 0 else 0.0)


def monte_carlo():
    rows = []
    for d in DEPTHS:
        for c in CLUTTER_LEVELS:
            for late in HINT_LATE:
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(seed)
                    clutter = clutter_field(rng, d, c) if c > 0 else None
                    noise = rng.normal(0.0, 10 ** (-40 / 20), (N_S, N_TR))
                    data, tau = base_section(d, clutter=clutter, noise=noise)
                    rg = radargram(data)
                    inputs = proto.est_mod.prepare_depth_inputs(rg, proto.CFG)
                    k_true = true_apex_sample(inputs, 2.0 / V * d)
                    hint = ApexHint(trace=J_A, sample=k_true + late * P / DT)
                    for name in VARIANTS:
                        rec, _ = run_variant(name, rg, inputs, hint, 2.0 / V * d)
                        rec.update(d_m=d, c=c, hint_late_P=late, seed=seed, variant=name)
                        rows.append(rec)
        print(f"depth {d} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DOCS / "m29c_track_repro_mc.csv", index=False, float_format="%.4f")
    g = df.groupby(["d_m", "c", "hint_late_P", "variant"])
    summ = g.agg(
        n=("seed", "size"),
        fail_insufficient=("enough", lambda s: float(1 - s.mean())),
        qc_in=("verdict", lambda s: float((s == "in").mean())),
        qc_unknown=("verdict", lambda s: float((s == "unknown").mean())),
        qc_out=("verdict", lambda s: float((s == "out").mean())),
        r_med=("r", "median"),
        r_q10=("r", lambda s: float(s.quantile(0.1))),
        r_q90=("r", lambda s: float(s.quantile(0.9))),
        apex_abs_dtr_med=("apex_dtr", lambda s: float(s.abs().median())),
        t0_err_med_ns=("t0_fixv_minus_true_ns", "median"),
        t0_err_p90_abs_ns=("t0_fixv_minus_true_ns", lambda s: float(s.abs().quantile(0.9))),
    ).reset_index()
    summ.to_csv(DOCS / "m29c_track_repro_mc_summary.csv", index=False, float_format="%.4f")
    return summ


def figure(summ):
    colors = {"v18": "#e34948", "med3": "#2a78d6", "descent": "#c9a227", "descent+med3": "#0b3d7a", "step": "#3fb08a"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True, sharex=True, sharey=True)
    for i, d in enumerate(DEPTHS):
        for j, late in enumerate(HINT_LATE):
            ax = axes[i, j]
            for name in VARIANTS:
                s = summ[(summ.d_m == d) & (summ.hint_late_P == late) & (summ.variant == name)]
                ax.plot(s.c, s.fail_insufficient, marker="o", ms=3, lw=1.2, color=colors[name], label=name)
            ax.set_title(f"d {d} m, hint {'at apex' if late == 0 else f'late {late} P'}", fontsize=9)
            ax.grid(color="#eeeeee")
            ax.set_ylim(-0.02, 1.0)
            if i == 1:
                ax.set_xlabel("clutter RMS / apex amplitude c", fontsize=8)
            if j == 0:
                ax.set_ylabel("insufficient_picks rate", fontsize=8)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(
        f"Synthetic reproduction (500 MHz, dt 0.098 ns, point-scatterer clutter, {N_SEEDS} seeds per point): "
        "tracking failure rate by rule", fontsize=9,
    )
    fig.savefig(DOCS / "m29c_track_repro.png", dpi=120)
    plt.close(fig)


def main() -> None:
    det = deterministic()
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.max_colwidth", 120):
        print(det.to_string(index=False))
    if "--det-only" in sys.argv:
        return
    summ = monte_carlo()
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 300):
        print(summ.to_string(index=False))
    figure(summ)


if __name__ == "__main__":
    main()
