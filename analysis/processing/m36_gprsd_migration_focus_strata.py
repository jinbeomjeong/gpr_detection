"""M3-6 D2m post-processing (report only, defined after seeing the D2m histogram; disclosed).

Splits the D2m windows by focus gain (max/min varimax over the g grid; high gain = a window whose
image changes strongly with g, i.e. diffraction-rich) and reports dz = dx/g_focus per stratum for
tau_z = 0, -40, 17. Edge windows (argmax on a grid end) are counted separately.
Output: analysis/processing/m36_gprsd_migration_focus_strata.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from m36_gprsd_common import DX_M, OUT_DIR, ROADS, qstats, road_bootstrap, write_json  # noqa: E402


def main() -> None:
    d = json.loads((OUT_DIR / "m36_gprsd_migration_focus.json").read_text(encoding="utf-8"))
    out = {}
    for tz in ("0", "-40", "17"):
        w = d["windows"]
        gain = np.array([x[f"focus_gain_tz{tz}"] for x in w])
        q50, q75 = np.percentile(gain, [50, 75])
        strata = {"gain_ge_p75": gain >= q75, "gain_p50_p75": (gain >= q50) & (gain < q75),
                  "gain_lt_p50": gain < q50}
        res = {"gain_p50": float(q50), "gain_p75": float(q75)}
        for name, sel in strata.items():
            rows = [x for x, s in zip(w, sel) if s]
            inner = [x for x in rows if not x[f"edge_tz{tz}"]]
            edge_lo = sum(1 for x in rows if x[f"edge_tz{tz}"] and x[f"g_tz{tz}"] < 1.0)
            edge_hi = sum(1 for x in rows if x[f"edge_tz{tz}"] and x[f"g_tz{tz}"] >= 1.0)
            by_road = {r: np.array([DX_M / x[f"g_tz{tz}"] for x in inner if x["road"] == r]) for r in ROADS}
            res[name] = {"n": len(rows), "n_interior": len(inner),
                         "n_edge_flat_(g_min)": edge_lo, "n_edge_steep_(g_max)": edge_hi,
                         "dz": qstats([DX_M / x[f"g_tz{tz}"] for x in inner], 5),
                         "dz_bootstrap": road_bootstrap(by_road) if len(inner) >= 4 else None}
        out[tz] = res
    p = write_json("m36_gprsd_migration_focus_strata.json", out)
    print(json.dumps({"status": "ok", "out": str(p)}))


if __name__ == "__main__":
    main()
