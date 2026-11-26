"""M3-6 figures from the D1a / D2 / D3 json outputs (no data re-processing).

Outputs (docs/processing/):
  m36_gprsd_period.png      D1a spectra by road (stored / moving-median removed), D1b histograms
  m36_gprsd_diffractors.png D2 (strict QC) / m36_gprsd_diffractors_planqc.png (plan QC) example fits (outside label windows only), tau_z profile, dz vs tau_z,
                            dz_i vs apex sample
  m36_gprsd_zero.png        D3a/D3b mean traces, early RMS, late-time RMS (gain trend, D6 (b))
Palette: dataviz default categorical slots 1-4 (validated light mode; legend + labels used).
Inputs: public json in analysis/processing/, except the two D2 files, which are read from the
local full copies in data/derived/m36_gprsd/ (per-diffractor tables and picks, git-ignored).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from m36_gprsd_common import FIG_DIR, OUT_DIR, ROADS, load_full_json  # noqa: E402

COL = dict(zip(ROADS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def load(name):
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


def period_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(4, 128)
    t = [5, 8, 10, 14.6, 20, 30, 51.2, 80, 128]
    ax.set_xticks(t)
    ax.set_xticklabels([str(x) for x in t])
    ax.minorticks_off()


def fig_period(per, dif):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    x = per["spectrum_period_axis"]
    for r in ROADS:
        s = per["spectra_by_road"][r]
        axs[0].plot(x, s["raw_mean_norm"], color=COL[r], lw=1.5, label=f"{r} (n={per['n_windows_by_road'][r]})")
        axs[1].plot(x, s["bg_removed_mean_norm"], color=COL[r], lw=1.5, label=r)
    axs[0].set_title("D1a stored amplitude: trace-mean |rfft| (window max = 1)", loc="left")
    axs[1].set_title("D1a after moving median (101 traces) removed", loc="left")
    for ax in axs[:2]:
        period_axis(ax)
        ax.set_xlabel("period (samples)")
        ax.set_ylabel("normalised amplitude")
    axs[0].legend(fontsize=7, loc="upper left")
    d = [v["gated_spec_period"] for v in dif["diffractors"] if v.get("gated_spec_period")]
    z = [v["zc_period"] for v in dif["diffractors"] if v.get("zc_period")]
    bins = np.arange(10, 54, 2)
    axs[2].hist(d, bins=bins, color="#2a78d6", alpha=0.85, label=f"gated spectrum peak (primary), p50 {np.median(d):.1f}",
                rwidth=0.9)
    axs[2].hist(z, bins=bins, histtype="step", color=INK, lw=1.5, label=f"2 x zero-crossing interval, p50 {np.median(z):.1f}")
    axs[2].set_title(f"D1b apex A-scan period (strict QC, n={len(d)})", loc="left")
    axs[2].set_xlabel("period (samples)")
    axs[2].set_ylabel("count")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m36_gprsd_period.png", dpi=130)
    plt.close(fig)


def fig_diff(dif, fname, tag):
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4)
    ds = dif["diffractors"]
    picks = dif["picks"]
    order = np.argsort([-d["apex_env"] for d in ds])
    tz_hat = dif["D2"]["tau_z_hat"]
    for k, i in enumerate(order[:4]):
        ax = fig.add_subplot(gs[0, k])
        d, p = ds[i], picks[i]
        u = np.asarray(p["u"])
        t = np.asarray(p["t"])
        ax.plot(u - d["u0"], t, "o", ms=3, color="#2a78d6", label="picks")
        uu = np.linspace(u.min(), u.max(), 200)
        ax.plot(uu - d["u0"], np.sqrt(d["h"] ** 2 + ((uu - d["u0"]) * d["g"]) ** 2), color=INK, lw=1.2,
                label=f"fit tau_z=0: dz {d['dz']:.4f} m")
        ax.invert_yaxis()
        ax.set_title(f"{d['file_id']} [{tag}] (outside label windows)", loc="left", fontsize=7)
        ax.set_xlabel("trace - u0")
        ax.set_ylabel("sample")
        ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[1, 0:2])
    grid0 = dif["tauz_grid"]
    tz = np.arange(grid0[0], grid0[1] + 0.5, 1.0)
    for r, ls in (("0.0", "-"), ("0.05", "--"), ("0.15", ":")):
        tot = np.array([np.nan if v is None else v for v in dif["profile_total_by_radius"][r]])
        ax.plot(tz, tot - np.nanmin(tot), ls=ls, color=INK, lw=1.4, label=f"R = {r} m")
    lo, hi = dif["D2"]["tau_z_ci"]["road_cluster"]["ci95"]
    ax.axvspan(lo, hi, color="#2a78d6", alpha=0.12, label="road-bootstrap 95 % CI (R = 0)")
    ax.axvline(tz_hat, color="#2a78d6", lw=1.2)
    ax.text(tz_hat, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, f" tau_z {tz_hat:.1f}", color=INK, fontsize=7)
    ax.set_xlim(-80, 25)
    ax.set_ylim(0, None)
    ax.set_xlabel("common zero tau_z (samples)")
    ax.set_ylabel("sum of mean soft_l1 cost - min")
    ax.set_title(f"D2 tau_z profile ({len(ds)} diffractors; infeasible above min apex {dif['D2']['min_tau_a']:.1f})",
                 loc="left")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 2])
    md = [(a, b) for a, b in dif["profile_dz_median_by_tauz"] if b is not None]
    ax.plot([a for a, _ in md], [b for _, b in md], color=INK, lw=1.5)
    ax.axhline(0.012, color="#eb6834", lw=1.2, ls="--")
    ax.text(-78, 0.0123, "provider dz 0.012", color=INK, fontsize=7)
    ax.set_xlabel("tau_z (samples)")
    ax.set_ylabel("median dz (m/sample)")
    ax.set_title("D2s median dz vs assumed zero", loc="left")
    ax = fig.add_subplot(gs[1, 3])
    for r in ROADS:
        sel = [d for d in ds if d["road"] == r]
        ax.plot([d["tau_a"] for d in sel], [d["dz"] for d in sel], "o", ms=4, color=COL[r], label=f"{r} ({len(sel)})",
                mec="white", mew=0.5)
    ax.axhline(0.012, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("apex sample (tau_z = 0)")
    ax.set_ylabel("dz (m/sample), tau_z = 0")
    ax.set_title("dz_i vs apex sample", loc="left")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


def fig_zero(zero):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for fid, x in zero["lines"].items():
        r = x["road"]
        lab = r if fid.endswith("_1") else None
        axs[0].plot(np.asarray(x["mean_trace_0_60"]) / x["line_rms"], color=COL[r], lw=1.3, label=lab)
        axs[1].plot(np.asarray(x["rms_samples_0_40"]) / x["line_rms"], color=COL[r], lw=1.3, label=lab)
        prof = np.asarray(x["rms_profile_bg_removed_every5"])
        axs[2].plot(np.arange(0, 321, 5), prof / prof[40:56].mean(), color=COL[r], lw=1.3, label=lab)
    axs[0].set_title("D3a/D3b whole-line mean trace / line RMS", loc="left")
    axs[0].set_xlabel("sample")
    axs[1].set_title("D3a RMS per sample / line RMS (samples 0-40)", loc="left")
    axs[1].set_xlabel("sample")
    axs[1].set_ylim(0, None)
    axs[2].set_title("(b) RMS after moving-median removal / mean RMS(200-275)", loc="left")
    axs[2].set_xlabel("sample")
    axs[2].set_ylim(0, None)
    for ax in axs:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m36_gprsd_zero.png", dpi=130)
    plt.close(fig)


def fig_mig(mig, dif, dif_plan):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0.005, 0.06, 23)
    for r in ROADS:
        v = [0.03 / w["g_tz0"] for w in mig["windows"] if w["road"] == r and not w["edge_tz0"]]
        axs[0].hist(v, bins=bins, histtype="step", lw=1.6, color=COL[r], label=f"{r} ({len(v)})")
    axs[0].axvline(0.012, color=MUTED, ls="--", lw=1)
    axs[0].text(0.0123, axs[0].get_ylim()[1] * 0.92, "provider 0.012", fontsize=7, color=INK)
    axs[0].set_xlabel("dz = dx / g_focus (m/sample, dx 0.03, tau_z 0)")
    axs[0].set_ylabel("windows")
    axs[0].set_title("D2m migration-focus scan, grid windows outside label windows", loc="left")
    axs[0].legend(fontsize=7)
    sets = [("D2m focus (windows)", [0.03 / w["g_tz0"] for w in mig["windows"] if not w["edge_tz0"]]),
            ("D2 plan QC (picks)", [d["dz"] for d in dif_plan["diffractors"]]),
            ("D2 strict QC (picks)", [d["dz"] for d in dif["diffractors"]])]
    axs[1].boxplot([s for _, s in sets], orientation="horizontal", widths=0.5, medianprops={"color": INK},
                   flierprops={"markersize": 3})
    axs[1].set_yticks([1, 2, 3])
    axs[1].set_yticklabels([n for n, _ in sets])
    axs[1].axvline(0.012, color=MUTED, ls="--", lw=1)
    axs[1].set_xlabel("dz at tau_z = 0 (m/sample, dx 0.03)")
    axs[1].set_title("dz by method (tau_z = 0)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m36_gprsd_migration_focus.png", dpi=130)
    plt.close(fig)


def main() -> None:
    per, zero = load("m36_gprsd_period.json"), load("m36_gprsd_zero.json")
    # per-diffractor tables and picks are in the local full files only (data/derived/m36_gprsd/)
    dif = load_full_json("m36_gprsd_diffractor_fit_strict.json")
    dif_plan = load_full_json("m36_gprsd_diffractor_fit.json")
    fig_period(per, dif)
    fig_diff(dif, "m36_gprsd_diffractors.png", "strict QC")
    fig_diff(dif_plan, "m36_gprsd_diffractors_planqc.png", "plan QC")
    fig_zero(zero)
    fig_mig(load("m36_gprsd_migration_focus.json"), dif, dif_plan)
    print(json.dumps({"status": "ok", "figs": ["m36_gprsd_period.png", "m36_gprsd_diffractors.png",
                                                 "m36_gprsd_diffractors_planqc.png", "m36_gprsd_zero.png",
                                                 "m36_gprsd_migration_focus.png"]}))


if __name__ == "__main__":
    main()
