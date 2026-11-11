"""Figures for M2-10 (d) (already-seen 4TU data, exploration only).

Reads the outputs of m210d_event_selection.py and draws:
* docs/processing/m210d_mechanism_by_depth.png - non-pass mechanism per truth depth bin
* docs/processing/m210d_pairs_scatter.png      - main pairs, failed pairs by paired-event class
* docs/processing/m210d_bscan_examples.png     - three lines with truth depths and candidates

Display gain (t^2) is for the picture only; no time is picked from it.
Run: conda run -n gpr-dev python -I analysis/processing/m210d_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.signals import pick_signal  # noqa: E402

BASE = Path(r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015")
DOCS = ROOT / "docs" / "processing"
DATA = ROOT / "data" / "processing"
ID = {"activity_id": str, "path": str}
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

GROUPS = [
    ("pass", "통과(±0.30 m)", "#9b9a95"),
    ("ii", "ii 근처 추정 있음(공유·배정)", CAT[0]),
    ("sup", "i 묶음·지지 필터", CAT[1]),
    ("qcE", "i QC E(곡률 unknown)", CAT[2]),
    ("qcBD", "i QC B·D", CAT[3]),
    ("gen", "i 후보 생성(피크·수락·상한)", CAT[6]),
    ("win", "iv 기록 끝 아래 정답", CAT[7]),
]


def group_of(mech: str) -> str:
    if mech == "pass":
        return "pass"
    if mech.startswith("ii"):
        return "ii"
    if mech.startswith("i-sup"):
        return "sup"
    if mech.startswith("i-qcE"):
        return "qcE"
    if mech.startswith(("i-qcB", "i-qcD")):
        return "qcBD"
    if mech.startswith(("i-gen", "i-fit")):
        return "gen"
    return "win"


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def fig_mechanism(truth: pd.DataFrame) -> None:
    truth = truth.copy()
    truth["group"] = truth["mechanism"].map(group_of)
    edges = [0.0, 0.5, 0.8, 1.1, 1.4, 5.0]
    names = ["≤0.5", "0.5–0.8", "0.8–1.1", "1.1–1.4", ">1.4"]
    truth["bin"] = pd.cut(truth["gt_depth_m"], edges, labels=names)
    tab = pd.crosstab(truth["bin"], truth["group"]).reindex(columns=[g for g, _, _ in GROUPS], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = np.zeros(len(tab))
    y = np.arange(len(tab))[::-1]
    for g, label, color in GROUPS:
        vals = tab[g].to_numpy()
        ax.barh(y, vals, left=left, color=color, height=0.62, label=label, edgecolor="#fcfcfb", linewidth=1.5)
        for yi, l, v in zip(y, left, vals):
            if v >= 8:
                ax.text(l + v / 2, yi, str(v), ha="center", va="center", fontsize=8, color="white")
        left += vals
    for yi, tot, (b, row) in zip(y, left, tab.iterrows()):
        ax.text(tot + 2, yi, f"n={int(tot)}, 통과 {row['pass']}/{int(tot)}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"{n} m" for n in tab.index])
    ax.set_xlabel("정답 관로 수 (exclude_x, 관 상단 가정, 검출 영점)", color=INK2)
    ax.set_title("깊은 정답일수록 후보 생성 단계에서 잃고, 얕은·중간 깊이는 '같은 깊이 추정 공유'로 잃는다\n"
                 "4TU 주 결과 정답 388개의 손실 기작(이미 본 자료·탐색, 판정 아님)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_xlim(0, left.max() * 1.25)
    style(ax)
    fig.text(0.01, 0.005, "출처: docs/processing/m210d_truth_classes.csv (m210d_event_selection.py)", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(DOCS / "m210d_mechanism_by_depth.png", dpi=150)
    plt.close(fig)


def fig_scatter(truth: pd.DataFrame, fp: pd.DataFrame) -> None:
    paired = truth[truth["status"].isin(["pass", "failed"])]
    ok = paired[paired["status"] == "pass"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lim = [0, 2.6]
    xs = np.linspace(*lim, 10)
    ax.plot(xs, xs, color=INK2, lw=1)
    ax.fill_between(xs, xs - 0.3, xs + 0.3, color="#ecebe7", lw=0, zorder=0)
    ax.plot(xs, 0.58 * xs, color=INK2, lw=1, ls="--")
    ax.text(2.25, 0.58 * 2.25 - 0.12, "εr 하한 √(3/9)=0.58", fontsize=7.5, color=INK2, ha="center")
    ax.scatter(ok["gt_depth_m"], ok["est_depth_m"], s=14, color="#9b9a95", label=f"통과 짝 {len(ok)}", zorder=2)
    classes = [
        ("iii-a second", "iii-a 다른 정답 관(통과)의 둘째 사건", CAT[0]),
        ("iii-a'", "iii-a′ 다른 정답 관 근처(미통과)", CAT[6]),
        ("iii-b", "iii-b 정답표에 없는 깊이(≥0.25 m)", CAT[1]),
        ("iii-c", "iii-c 지표 근처(<0.25 m)", CAT[2]),
    ]
    for key, label, color in classes:
        sub = fp[fp["paired_event_class"].str.startswith(key)]
        ax.scatter(sub["gt_depth_m"], sub["est_depth_m"], s=36, color=color, edgecolor="#fcfcfb", linewidth=1,
                   label=f"{label} {len(sub)}", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("정답 깊이 d (m, 관 상단 가정)", color=INK2)
    ax.set_ylabel("짝지어진 추정 깊이 d̂ (m)", color=INK2)
    ax.set_title("실패 짝의 추정은 그 관의 얕은 오추정이 아니라\n다른 사건(다른 관·미기재 반사·지표 근처)이다",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    fig.text(0.01, 0.005, "출처: m210d_failed_pairs.csv · 이미 본 자료(4TU), 탐색", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(DOCS / "m210d_pairs_scatter.png", dpi=150)
    plt.close(fig)


EXAMPLES = [
    ("01.1", "Path9", "01.1 Path9 — 1.30 m 꼭짓점은 상한 6에서 잘리고, 얕은 꼭짓점이 깊은 정답과 짝"),
    ("010.20", "Path8", "010.20 Path8 — 깊은 정답(1.29–1.47 m) 근처 후보는 QC B(곡률 out)로 빠짐"),
    ("011.2", "Profile_6068", "011.2 Profile_6068 — 1.43–2.33 m 정답 근처에 검출 피크 없음(εr 17.4)"),
]


def fig_bscans(truth: pd.DataFrame, cand: pd.DataFrame, est: pd.DataFrame, gen: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (act, path, title) in zip(axes, EXAMPLES):
        rg = load_fourtu_radargram(BASE, act, path)
        inputs = prepare_depth_inputs(rg)
        t_z = inputs.zero.t_zero_ns
        v = velocity_from_eps(rg.eps_r_meta)
        env = pick_signal(inputs.branch.background_removed, "envelope")
        t = np.arange(env.shape[0]) * rg.dt_ns
        disp = env * np.clip(t - t_z, 0.5, None)[:, None] ** 2  # display gain only
        disp = disp / np.percentile(disp[t > t_z + 2], 99.5)
        d_axis = v * (t - t_z) / 2
        x = np.arange(env.shape[1]) * rg.dx_m
        ax.imshow(np.clip(disp, 0, 1), aspect="auto", cmap="Greys",
                  extent=[x[0], x[-1], d_axis[-1], d_axis[0]], interpolation="nearest")
        gt = truth[truth["activity_id"] == act].sort_values("gt_depth_m")
        seen: dict[float, int] = {}
        for _, r in gt.iterrows():
            color = {"pass": "#9b9a95", "failed": CAT[7], "missed": CAT[3]}[r["status"]]
            ax.axhline(r["gt_depth_m"], color=color, lw=1.2, ls="--")
            key = round(r["gt_depth_m"], 2)
            k = seen.get(key, 0)
            seen[key] = k + 1
            ax.text(x[-1], r["gt_depth_m"] + 0.07 * k, f" {r['code']} {r['gt_depth_m']:.2f} {r['status']}",
                    fontsize=7, color=color, va="center", ha="left", clip_on=False)
        g = gen[(gen["activity_id"] == act) & (gen["path"] == path) & (gen["reason"] == "nms_or_cap")]
        ax.scatter(g["trace"].astype(float) * rg.dx_m, g["depth_nom_m"], marker="^", s=22, color=CAT[6],
                   label="수락됐으나 NMS·상한 6에서 잘림", zorder=3)
        c = cand[(cand["activity_id"] == act) & (cand["path"] == path)]
        qx = c["qc_excluded"].astype(str) == "True"
        dep = pd.to_numeric(c["depth_top_m"], errors="coerce")
        xs = c["candidate_trace"].astype(float) * rg.dx_m
        ax.scatter(xs[qx], dep[qx], marker="x", s=30, color=CAT[1], label="QC 제외 후보", zorder=3)
        ax.scatter(xs[~qx], dep[~qx], marker="o", s=40, facecolor="none", edgecolor=CAT[0], linewidth=1.6,
                   label="QC 통과 후보", zorder=3)
        sup = est[(est["activity_id"] == act) & (est["kept_by_support"].astype(str) == "True")]
        for d in sup["depth_m"]:
            ax.plot([x[0], x[0] + 0.25], [d, d], color=CAT[0], lw=3)
        ax.set_ylim(min(2.6, d_axis[-1]), 0)
        ax.set_xlabel("측선 위치 (m, 트레이스 × 0.02)", color=INK2)
        ax.set_ylabel(f"깊이 (m, 메타 εr {rg.eps_r_meta:g}, 검출 영점)", color=INK2)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.suptitle("대표 측선: 정답 깊이(점선; 회색 통과·빨강 실패 짝·노랑 누락)와 꼭짓점 후보, 왼쪽 굵은 눈금 = 활동 지지 추정 깊이"
                 " (표시용 t² 이득, 이미 본 자료)", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(DOCS / "m210d_bscan_examples.png", dpi=130)
    plt.close(fig)


def main() -> None:
    truth = pd.read_csv(DATA / "m210d_truth_full.csv", dtype=ID)
    fp = pd.read_csv(DOCS / "m210d_failed_pairs.csv", dtype=ID)
    cand = pd.read_csv(DOCS / "depth_4tu_v2_candidates.csv", dtype=ID, keep_default_na=False)
    est = pd.read_csv(DATA / "m210d_estimates.csv", dtype=ID)
    fig_mechanism(truth)
    fig_scatter(truth, fp)
    gen = pd.read_csv(DATA / "m210d_generator_items.csv", dtype=ID)
    fig_bscans(truth, cand, est, gen)
    sh = fp[fp["side"] == "shallow"]
    print("failed shallow: est median", sh["est_depth_m"].median(), "gt median", sh["gt_depth_m"].median(),
          "ratio of medians", round(sh["est_depth_m"].median() / sh["gt_depth_m"].median(), 3))


if __name__ == "__main__":
    sys.exit(main())
