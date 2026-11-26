"""Summarize m33b_pp_alarm_sim.json into plain-text tables (design numbers only)."""

import json
import math
import sys

src, out = sys.argv[1], sys.argv[2]
res = json.load(open(src, encoding="utf-8"))
reps = res["reps"]
lines = []


def mcse(p):
    return math.sqrt(max(p * (1 - p), 1e-12) / reps)


def cell(var, n, r0, kind, delta):
    for a in res["alarm"]:
        if a["var"] == var and a["n"] == n and abs(a["r0"] - r0) < 1e-12 and a["kind"] == kind and abs(a["delta_pp"] - delta) < 1e-9:
            return a
    return None


def mde80(var, n, r0, kind, key):
    pts = []
    for a in res["alarm"]:
        if a["var"] == var and a["n"] == n and abs(a["r0"] - r0) < 1e-12 and a["kind"] == kind and not a["unreachable"]:
            pts.append((-a["delta_pp"], a["p_alarm"][key]))
    pts.sort()
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 < 0.8 <= y1:
            return x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)
    if pts and pts[0][1] >= 0.8:
        return f"<={pts[0][0]:g}"
    return "unreached(max %.2f at %g)" % (max(p[1] for p in pts), pts[-1][0]) if pts else "none"


deltas = [-0.5, -1.0, -1.5, -2.0, -3.0, -5.0, -7.5, -10.0]
lines.append(f"reps={reps} seed={res['seed']} designs={json.dumps(res['designs'])}")
lines.append(f"selfcheck={json.dumps(res['selfcheck'])}")
for key in ["abs1|m0_ge25", "abs1|none", "rel10|m0_ge25", "gt_lost3|none", "abs1|gt10_trip3"]:
    lines.append("")
    lines.append(f"== P(alarm) key={key} var=base h=0.9 (loss only); bal1/bal2 = balanced gross 1/2 %p (true delta 0)")
    lines.append("n  R0   P(info25) | bal1  bal2 | " + " ".join(f"{d:>6g}" for d in deltas) + " | MDE80")
    for n in [50, 100, 150, 190]:
        for r0 in [0.02, 0.05, 0.10, 0.20, 0.40]:
            row = []
            pinfo = None
            for b in ["balanced_gross1", "balanced_gross2"]:
                c = cell("base", n, r0, b, 0.0)
                row.append(f"{c['p_alarm'][key]:.3f}" if c else "  -  ")
                if c:
                    pinfo = c["p_info"]["m0_ge25"]
            vals = []
            for d in deltas:
                c = cell("base", n, r0, "loss_only_h0.9", d)
                if c is None or c.get("unreachable"):
                    vals.append("   unr")
                else:
                    vals.append(f"{c['p_alarm'][key]:6.3f}")
                    pinfo = c["p_info"]["m0_ge25"]
            m = mde80("base", n, r0, "loss_only_h0.9", key)
            ms = f"{m:.2f}" if isinstance(m, float) else m
            lines.append(f"{n:<3d} {r0:.2f} {pinfo if pinfo is None else round(pinfo, 3)} | {' '.join(row)} | {' '.join(vals)} | {ms}")

lines.append("")
lines.append("== Sensitivity abs1|m0_ge25: MDE80 (%p) by var scenario / h")
for var in ["base", "weak_road", "strong_road", "loose_lines"]:
    for kind in (["loss_only_h0.9", "loss_only_h0.5"] if var == "base" else ["loss_only_h0.9"]):
        for n in [50, 190]:
            for r0 in [0.05, 0.20]:
                m = mde80(var, n, r0, kind, "abs1|m0_ge25")
                ms = f"{m:.2f}" if isinstance(m, float) else m
                c1 = cell(var, n, r0, kind, -0.5)
                c2 = cell(var, n, r0, kind, -2.0)
                b1 = cell(var, n, r0, "balanced_gross1", 0.0)
                lines.append(
                    f"{var:12s} {kind:15s} n={n:<3d} R0={r0:.2f} MDE80={ms:>8s} "
                    f"P(-0.5)={c1['p_alarm']['abs1|m0_ge25'] if c1 and not c1.get('unreachable') else '-'} "
                    f"P(-2)={c2['p_alarm']['abs1|m0_ge25'] if c2 and not c2.get('unreachable') else '-'} "
                    f"P(bal1)={b1['p_alarm']['abs1|m0_ge25'] if b1 else '-'} "
                    f"sd_dR(-2)={c2['sd_dR_pp'] if c2 and not c2.get('unreachable') else '-'}"
                )

lines.append("")
lines.append("== P(info) variants, base, delta=-2 cell (info does not depend on delta)")
for n in [50, 100, 150, 190]:
    for r0 in [0.02, 0.05, 0.10, 0.20, 0.40]:
        c = cell("base", n, r0, "balanced_gross1", 0.0) or cell("base", n, r0, "loss_only_h0.9", -0.5)
        if c:
            lines.append(f"n={n:<3d} R0={r0:.2f} " + " ".join(f"{k}={v:.3f}" for k, v in c["p_info"].items()))

lines.append("")
lines.append(f"== CI coverage (nominal 0.95) cov_reps={res['cov_reps']} n_boot={res['n_boot']}")
for c in res["coverage"]:
    lines.append(f"-- var={c['var']} n={c['n_actual']} R0={c['r0']} mean_est={c['mean_est']:.4f}")
    for k, v in c.items():
        if isinstance(v, dict):
            lines.append(f"   {k:26s} superpop={v['cover_superpop']:.3f} roadcond={v['cover_roadcond']:.3f} width={v['mean_width_pp']:.2f}pp")

open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(json.dumps({"status": "ok", "out": out}))
