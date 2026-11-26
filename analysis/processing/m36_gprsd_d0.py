"""M3-6 D0: input schema re-check summary.

Collects the tools/pyinspect/data_inspect.py reports (run beforehand with --expect,
schemas: npy [321, None] float32; json keys dt_ns_hypA_321, dt_ns_hypB_380, trace_dx_m,
depth_per_sample_m_rough, n_samples, n_traces) from the folder given as argv[1] and adds the
scale fields of each line json.

Usage: python -I analysis/processing/m36_gprsd_d0.py <report_dir>
Output: analysis/processing/m36_gprsd_d0.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m36_gprsd_common import line_ids, line_meta, write_json  # noqa: E402


def main() -> None:
    rep = Path(sys.argv[1])
    rows = []
    n_find = 0
    for fid in line_ids():
        rn = json.loads((rep / f"{fid}.npy.json").read_text(encoding="utf-8"))
        rj = json.loads((rep / f"{fid}.json.json").read_text(encoding="utf-8"))
        fn = rn["expect"]["findings"] + rj["expect"]["findings"]
        n_find += len(fn)
        m = line_meta(fid)
        st = rn["tree"]["stats"]
        rows.append({
            "file_id": fid, "shape": rn["tree"]["shape"], "dtype": rn["tree"]["dtype"],
            "nan": st["nan_count"], "inf": st["inf_count"],
            "min": round(st["min"], 1), "max": round(st["max"], 1),
            "n_samples_json": m["n_samples"], "n_traces_json": m["n_traces"],
            "shape_matches_json": rn["tree"]["shape"] == [m["n_samples"], m["n_traces"]],
            "dt_ns_hypA_321": m["dt_ns_hypA_321"], "dt_ns_hypB_380": m["dt_ns_hypB_380"],
            "trace_dx_m": m["trace_dx_m"], "depth_per_sample_m_rough": m["depth_per_sample_m_rough"],
            "time_window_doc_ns": m["time_window_doc_ns"], "findings": fn,
            "npy_notes": rn["tree"].get("notes", []),
        })
    uniq = {k: sorted({json.dumps(r[k]) for r in rows}) for k in
            ("dtype", "dt_ns_hypA_321", "dt_ns_hypB_380", "trace_dx_m",
             "depth_per_sample_m_rough", "time_window_doc_ns")}
    out = {"tool": "~/.claude/tools/pyinspect/data_inspect.py --expect (python-dev -I -B)",
           "n_lines": len(rows), "n_findings": n_find,
           "all_shape_match_json": all(r["shape_matches_json"] for r in rows),
           "unique_values": uniq, "lines": rows}
    p = write_json("m36_gprsd_d0.json", out)
    print(json.dumps({"status": "ok", "n_lines": len(rows), "n_findings": n_find, "out": str(p)}))


if __name__ == "__main__":
    main()
