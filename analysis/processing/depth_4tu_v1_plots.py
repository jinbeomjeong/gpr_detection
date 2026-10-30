"""Figures of the M2-8 4TU depth check (committed csv only; no radargram is read).

Main combination everywhere: gt_set exclude_x, top assumption, detected zero,
variant main (support filter), unless a figure says otherwise.

Figures (docs/processing/):
1. depth_4tu_v1_scatter.png    estimate vs truth depth of the matched pairs, y = x and +-0.30 m band
2. depth_4tu_v1_errors.png     error distribution (estimate - truth), +-0.30 m limits, bias
3. depth_4tu_v1_passb.png      pass rate B, observed vs random baseline (mean, 2.5-97.5 %), all + sites
4. depth_4tu_v1_causes.png     (a) apex re-seed time shift, (b) depth ECDF: truth / candidate / re-seeded
5. depth_4tu_v1_factors.png    factor swap of the diagnosis (observed vs baseline mean)

Palette: reference categorical slots 1-3 (blue, orange, aqua), validated all-pairs
light (dataviz validate_palette.js: CVD dE 9.2, normal 24.0; aqua < 3:1 -> direct
labels + legend). Korean labels in Malgun Gothic.
Run:  conda run -n gpr-dev python -I analysis/processing/depth_4tu_v1_plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "docs" / "processing"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
PASS_M = 0.30

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def main_pairs(matches: pd.DataFrame, variant: str = "main") -> pd.DataFrame:
    sel = ((matches["gt_set"] == "exclude_x") & (matches["assumption"] == "top")
           & (matches["zero_variant"] == "detected") & (matches["variant"] == variant))
    return matches[sel]


def fig_scatter(pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    lim = (0, max(2.2, float(pairs[["est_depth_m", "gt_depth_m"]].max().max()) + 0.1))
    x = np.linspace(*lim, 2)
    ax.fill_between(x, x - PASS_M, x + PASS_M, color=GRID, lw=0, label="±0.30 m (제186조)")
    ax.plot(x, x, color=MUTED, lw=1.0, ls="--", label="y = x")
    inside = pairs["within_30cm"].astype(str) == "True"
    ax.scatter(pairs.loc[inside, "gt_depth_m"], pairs.loc[inside, "est_depth_m"], s=22, color=BLUE,
               edgecolor=SURFACE, lw=0.8, label=f"±0.30 m 안 ({inside.sum()})")
    ax.scatter(pairs.loc[~inside, "gt_depth_m"], pairs.loc[~inside, "est_depth_m"], s=22, color=ORANGE,
               edgecolor=SURFACE, lw=0.8, marker="D", label=f"±0.30 m 밖 ({(~inside).sum()})")
    ax.set(xlim=lim, ylim=lim, xlabel="정답 깊이 (m, 상단 가정)", ylabel="추정 깊이 (m)")
    ax.set_title(f"추정 vs 정답 — 짝 {len(pairs)}개, 대부분 y = x 아래(얕게 추정)", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(DOC / "depth_4tu_v1_scatter.png", dpi=150)


def fig_errors(pairs: pd.DataFrame) -> None:
    e = pairs["error_m"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    bins = np.arange(-1.6, 0.81, 0.05)
    ax.axvspan(-PASS_M, PASS_M, color=GRID, lw=0)
    ax.hist(e, bins=bins, color=BLUE, edgecolor=SURFACE, lw=0.8)
    ax.axvline(e.mean(), color=ORANGE, lw=2)
    ax.text(e.mean(), ax.get_ylim()[1] * 0.95, f" bias {e.mean():+.3f} m", color=INK, va="top", ha="right", fontsize=8,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.text(PASS_M, ax.get_ylim()[1] * 0.80, " ±0.30 m 경계", color=INK2, ha="left", fontsize=8)
    ax.set(xlabel="오차 = 추정 - 정답 (m, 음수 = 얕게)", ylabel="짝 수")
    ax.set_title(f"오차 분포 — 짝 {len(e)}개, MAE {np.abs(e).mean():.3f} m, ±0.30 m 안 "
                 f"{np.mean(np.abs(e) <= PASS_M + 1e-12):.0%}", color=INK, fontsize=10)
    fig.savefig(DOC / "depth_4tu_v1_errors.png", dpi=150)


def fig_passb(summary: pd.DataFrame) -> None:
    sel = ((summary["gt_set"] == "exclude_x") & (summary["assumption"] == "top")
           & (summary["zero_variant"] == "detected") & (summary["variant"] == "main"))
    s = summary[sel].copy()
    s["order"] = s["scope"].map(lambda v: -1 if v == "all" else int(v.split("_")[1]))
    s = s.sort_values("order")
    labels = ["전체" if v == "all" else f"현장 {v.split('_')[1]}" for v in s["scope"]]
    y = np.arange(len(s))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.hlines(y, s["baseline_pass_b_p025"], s["baseline_pass_b_p975"], color=ORANGE, lw=6, alpha=0.35,
              label="무작위 기준선 2.5–97.5 %")
    ax.scatter(s["baseline_pass_b_mean"], y, color=ORANGE, marker="|", s=160, lw=2, label="기준선 평균")
    ax.scatter(s["pass_b"], y, color=BLUE, s=40, zorder=3, edgecolor=SURFACE, lw=0.8, label="관측 판정률 B")
    for yi, row in zip(y, s.itertuples(), strict=True):
        ax.text(1.02, yi, f"{row.pass_b:.2f} / {row.baseline_pass_b_mean:.2f}  (정답 {row.n_gt})",
                va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("판정률 B = ±0.30 m 안 짝 수 / 정답 수")
    ax.set_title("관측 vs 무작위 기준선 — 전체와 대부분 현장에서 기준선보다 낮음", color=INK, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5)
    fig.savefig(DOC / "depth_4tu_v1_passb.png", dpi=150, bbox_inches="tight")


def ecdf(values):
    v = np.sort(np.asarray(values, float))
    return v, np.arange(1, len(v) + 1) / len(v)


def fig_causes(cand: pd.DataFrame, apex: pd.DataFrame, lines: pd.DataFrame, gt_depths) -> None:
    dt = dict(zip(zip(lines["activity_id"], lines["path"]), lines["dt_ns"], strict=True))
    ok = cand[cand["status"] == "ok"].merge(
        apex[["activity_id", "path", "trace", "sample", "tau_ns"]],
        left_on=["activity_id", "path", "candidate_trace", "candidate_sample"],
        right_on=["activity_id", "path", "trace", "sample"], how="left")
    ok["dt"] = [dt[(a, p)] for a, p in zip(ok["activity_id"], ok["path"], strict=True)]
    shift = (ok["apex_sample"] - ok["candidate_sample"]) * ok["dt"]
    v = 0.299792458 / np.sqrt(ok["eps_r_used"])
    cand_depth = v * ok["tau_ns"] / 2
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.hist(shift.clip(-5, 3), bins=np.arange(-5, 3.01, 0.1), color=BLUE, edgecolor=SURFACE, lw=0.5)
    ax.axvline(-3.0, color=ORANGE, lw=1.5, ls="--")
    ax.text(-3.0, ax.get_ylim()[1] * 0.92, " 박스 위 여유 1.5/f_c = 3 ns", color=INK, fontsize=8)
    ax.set(xlabel="재선정 꼭짓점 시각 - 꼭짓점 후보 시각 (ns)", ylabel="꼭짓점 후보 수")
    ax.set_title(f"(a) 재선정이 후보의 {np.mean(shift < -0.5):.0%}를 약 3 ns 위로 옮김", color=INK, fontsize=10)
    ax = axes[1]
    for vals, color, name in ((gt_depths, MUTED, "정답"), (cand_depth.dropna(), AQUA, "꼭짓점 후보 시각 그대로"),
                              (ok["depth_top_m"], BLUE, "재선정 뒤(주 결과 경로)")):
        xs, ys = ecdf(vals)
        ax.plot(xs, ys, color=color, lw=2, label=f"{name} (중앙 {np.median(vals):.2f} m)")
    ax.set(xlim=(0, 2.0), ylim=(0, 1), xlabel="깊이 (m)", ylabel="누적 비율")
    ax.set_title("(b) 후보부터 얕고, 재선정이 더 얕게 만든다", color=INK, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(DOC / "depth_4tu_v1_causes.png", dpi=150)


def fig_factors(diag: pd.DataFrame) -> None:
    names = {("reseeded (as run)", "single_linkage (as run)"): "실행 그대로\n(재선정 + 단일 연결)",
             ("reseeded (as run)", "span_limited_0.10"): "재선정\n+ 폭 제한 묶음",
             ("candidate_direct", "single_linkage (as run)"): "후보 시각 그대로\n+ 단일 연결",
             ("candidate_direct", "span_limited_0.10"): "후보 시각 그대로\n+ 폭 제한 묶음"}
    labels = [names[(r.depths, r.clusters)] for r in diag.itertuples()]
    y = np.arange(len(diag))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.hlines(y, diag["pass_b"], diag["baseline_pass_b_mean"], color=GRID, lw=3)
    ax.scatter(diag["baseline_pass_b_mean"], y, color=ORANGE, marker="|", s=160, lw=2, label="기준선 평균(같은 추정 수)")
    ax.scatter(diag["pass_b"], y, color=BLUE, s=40, zorder=3, edgecolor=SURFACE, lw=0.8, label="관측 판정률 B")
    for yi, row in zip(y, diag.itertuples(), strict=True):
        ax.text(1.02, yi, f"{row.pass_b:.2f} / {row.baseline_pass_b_mean:.2f}, 추정 {row.n_est}",
                va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("판정률 B (진단 전용, 기준선 200회)")
    ax.set_title("결함 두 개를 바꿔도 기준선보다 낮음 — 후보 자체가 얕은 쪽", color=INK, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    fig.savefig(DOC / "depth_4tu_v1_factors.png", dpi=150, bbox_inches="tight")


def main() -> None:
    matches = pd.read_csv(DOC / "depth_4tu_v1_matches.csv", dtype={"activity_id": str})
    summary = pd.read_csv(DOC / "depth_4tu_v1_summary.csv")
    cand = pd.read_csv(DOC / "depth_4tu_v1_candidates.csv", dtype={"activity_id": str})
    apex = pd.read_csv(DOC / "fourtu_apex_candidates_v1.csv", dtype={"activity_id": str})
    lines = pd.read_csv(DOC / "fourtu_lines_v1.csv", dtype={"activity_id": str})
    diag = pd.read_csv(DOC / "depth_4tu_v1_diagnose.csv")
    pairs = main_pairs(matches)
    fig_scatter(pairs)
    fig_errors(pairs)
    fig_passb(summary)
    # Truth depths: variant b (unclustered) matches 385 of the 388 truths, the closest
    # csv-only proxy of the full truth list.
    gt_depths = main_pairs(matches, "b")["gt_depth_m"]
    fig_causes(cand, apex, lines, gt_depths)
    fig_factors(diag)


if __name__ == "__main__":
    main()
