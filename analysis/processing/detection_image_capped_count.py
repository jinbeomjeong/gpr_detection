"""Count 4TU lines whose O_phys is set by the period cap (spec_detection_image v0.3.2, 4-4 item 9).

Same defaults as the spec: P 2.51 ns (dataset default), v from eps_r_used, a_max 1 m,
w_P 20 periods, O_min 64, O_cv 128. Uses actual sigma_x = N / W1. Metadata only.
"""
import csv
import math
from collections import Counter

ROWS = list(csv.DictReader(open(r"C:\workspace\gpr_detection\docs\processing\fourtu_lines_v1.csv", encoding="utf-8-sig")))
C = 0.299792458
CHECK = {("01.1", "Path1"), ("03.9", "Path5"), ("09.1", "Path2"), ("09.4", "Path1")}

for p_star in (4.06, 6.0):
    capped, o_used, eps_capped = 0, Counter(), Counter()
    for r in ROWS:
        dt, n, eps = float(r["dt_ns"]), int(r["n_traces"]), float(r["eps_r_used"])
        v = C / math.sqrt(eps)
        s_y = (2.51 / dt) / p_star
        s_x = s_y * (v * dt / 2) / 0.02
        w1 = max(1, round(n / s_x))
        mpp = (n / w1) * 0.02
        metres = 2.0 / mpp
        cap = 20.0 * p_star
        is_capped = metres > cap
        o_phys = max(64, math.ceil(min(metres, cap) - 1e-9))
        o = min(224, max(o_phys, 128))
        capped += is_capped
        o_used[o] += 1
        if is_capped:
            eps_capped[eps] += 1
        if (r["activity_id"], r["path"]) in CHECK:
            print(p_star, r["activity_id"], r["path"], "eps", eps, "metres_px", round(metres, 1), "capped", is_capped, "O", o)
    print("P*", p_star, "capped", capped, "of", len(ROWS), "O used", dict(o_used), "eps of capped", sorted(eps_capped.items()))
print("threshold v < 4/(20*2.51) =", round(4 / (20 * 2.51), 5), "-> eps >", round((C / (4 / (20 * 2.51))) ** 2, 2))
