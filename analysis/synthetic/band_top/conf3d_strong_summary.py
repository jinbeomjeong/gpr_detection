"""2D vs 3D comparison for the strong-layer confirmation (wet bed, steel plate).

  C:\\ProgramData\\miniforge3\\envs\\python-dev\\python.exe -I analysis/synthetic/band_top/conf3d_strong_summary.py

Reads export/<id>.json written by export.py --stage conf3d_strong (and the 2D W_PEC_A10).
shadow = pipe-only envelope max under the plate / pipe-only envelope max of the same pipe without plate (same grid).
plate_to_unshadowed = plate-only envelope max / unshadowed pipe-only envelope max.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent.parent / "data" / "synthetic_gprmax_band_top" / "export"
P = 2.0


def load(mid):
    p = EXP / f"{mid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    meta, seen = [], set()
    for p in sorted(HERE.glob("models_conf3d_strong*.json")):
        for m in json.loads(p.read_text(encoding="utf-8"))["models"]:
            if m["id"] not in seen:
                seen.add(m["id"])
                meta.append(m)
    rows = []
    for m in meta:
        if not m["pipe"] or m["role"] != "conf3d_strong":
            continue
        j = load(m["id"])
        if j is None:  # not run or not exported yet
            continue
        row = {"id": m["id"], "dim": m["dim"], "pipe": m["pipe"], "dipole": m.get("dipole"),
               "kind": m.get("strong_kind"), "plate_width_m": m.get("plate_width_m"),
               "plate_section": m.get("plate_section"), "plate_top_above_pipe_top_m": m.get("plate_top_above_pipe_top_m"),
               "ratio_layer_to_pipe": j["ratio_layer_to_pipe_measured"],
               "layer_dt_P": j["layer_dt_measured_P"],
               "t_top_minus_tau_v_ns": j["t_top_true_ns"] - j["tau_top_vertical_ns"],
               "pipe_only_env_max": j["pipe_only_env_max"], "layer_env_max": j["layer_only_env_max"]}
        ref_id = m.get("pipe_unshadowed")
        if ref_id:
            ref = load(ref_id)
            row["unshadowed_pipe_env_max"] = ref["pipe_only_env_max"]
            row["shadow_fraction"] = j["pipe_only_env_max"] / ref["pipe_only_env_max"]
            row["plate_to_unshadowed_pipe"] = j["layer_only_env_max"] / ref["pipe_only_env_max"]
            row["unshadowed_lag_ns"] = ref["t_top_true_ns"] - ref["tau_top_vertical_ns"]
        rows.append(row)
    w2 = load("W_PEC_A10")
    if w2:
        rows.append({"id": "W_PEC_A10", "dim": 2, "pipe": "pec", "dipole": "z (TMz)", "kind": "tuned_wet_bed",
                     "ratio_layer_to_pipe": w2["ratio_layer_to_pipe_measured"], "layer_dt_P": w2["layer_dt_measured_P"],
                     "t_top_minus_tau_v_ns": w2["t_top_true_ns"] - w2["tau_top_vertical_ns"],
                     "pipe_only_env_max": w2["pipe_only_env_max"], "layer_env_max": w2["layer_only_env_max"]})
    (EXP / "summary_conf3d_strong_compare.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        extra = ""
        if "shadow_fraction" in r:
            extra = (f" shadow={100 * r['shadow_fraction']:.1f}% plate/pipe0={r['plate_to_unshadowed_pipe']:.1f}"
                     f" lag0={r['unshadowed_lag_ns']:.3f}")
        print(f"{r['id']:<22} {r['dim']}D pol={r['dipole']} ratio={r['ratio_layer_to_pipe']:.2f} "
              f"layP={[round(x, 2) for x in r['layer_dt_P']]} lag={r['t_top_minus_tau_v_ns']:.3f}{extra}")


if __name__ == "__main__":
    main()
