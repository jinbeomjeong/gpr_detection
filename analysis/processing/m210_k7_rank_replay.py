"""M2-10 K7: offline replays of alternative ranking keys (development evidence, no judgement).

Reorders the accepted items of the unchanged generator and replays NMS + per-line
cap (``m210_k7_probe.replay``). Keys:

* ``cur``  : (-min(n_left, n_right), -snr)   (generator)
* ``b2``   : (-snr,)                          detection SNR only
* ``b3``   : (-E,)                            raw envelope at the item (S1g only;
                                              the 4TU item csv has no E)
* ``b4``   : (-(n_left + n_right), -snr)      total wing support

Inputs: S1g exports (already seen) and ``data/processing/m210d_generator_items.csv``
(4TU, already seen, exploration only).
Output: ``docs/processing/m210_k7_rank_replay.json``.

Run: conda run -n gpr-dev python -I analysis/processing/m210_k7_rank_replay.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))
sys.path.insert(0, str(ROOT / "scripts"))

import fourtu_apex_candidate_select as gen  # noqa: E402
import m210_k7_probe as probe  # noqa: E402
import m210_s1g_pipeline as s1g  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402

OUT = ROOT / "docs" / "processing" / "m210_k7_rank_replay.json"
KEYS = {
    "cur": lambda s: (-min(s["n_left"], s["n_right"]), -s["snr"]),
    "b2": lambda s: (-s["snr"],),
    "b3": lambda s: (-s["E"],),
    "b4": lambda s: (-(s["n_left"] + s["n_right"]), -s["snr"]),
}


def s1g_part():
    out = {}
    for model in ("S1G_C1", "S1G_D060", "S1G_D100"):
        rg = s1g.load_line(model, probe.S1G_ROOT)
        meta = json.loads((probe.S1G_ROOT / f"{model}.json").read_text(encoding="utf-8"))
        inputs = prepare_depth_inputs(rg, DepthConfig())
        line, kept, E = gen.apex_candidates_for_radargram(rg, inputs, act="S1G", path=model)
        acc = [dict(s) for s in line["_accepted_all"]]
        for s in acc:
            s["E"] = float(E[int(round(s["sample"])), s["trace"]])
        res = {}
        for name, key in KEYS.items():
            items = sorted(acc, key=key)
            kept_v, _ = probe.replay(items, rg.dx_m, rg.dt_ns, rg.f_c_hz, "current")
            per = []
            for p in meta["pipes"]:
                ja, ta = p["apex_trace_index"], p["apex"]["envelope_peak_ns"]
                per.append(
                    sum(abs(s["trace"] - ja) <= 3 and abs(s["t_ns"] - ta) <= 1.0 for s in kept_v)
                )
            res[name] = {
                "kept_in_window_per_pipe": per,
                "kept": [
                    {"trace": s["trace"], "t_ns": s["t_ns"], "snr": s["snr"],
                     "min_wing": min(s["n_left"], s["n_right"]), "E": round(s["E"], 3)}
                    for s in kept_v
                ],
            }
        out[model] = res
    return out


def fourtu_part():
    items = pd.read_csv(probe.ITEMS, dtype=probe.ID_COLUMNS)
    lines = pd.read_csv(probe.LINES, dtype=probe.ID_COLUMNS)
    dt_of = {(r.activity_id, r.path): float(r.dt_ns) for r in lines.itertuples()}
    acc = items[items["stage"].isin(["kept", "accepted"])]
    tot = {k: {"n_kept": 0, "lines_changed": 0, "new_far": 0, "lost_far": 0,
               "median_tau_kept": None} for k in KEYS if k != "b3"}
    taus = {k: [] for k in tot}
    for (act, path), g in acc.groupby(["activity_id", "path"], sort=False):
        dt = dt_of[(act, path)]
        rows = [
            {"trace": int(r.trace), "t_ns": round(float(r.sample) * dt, 3),
             "snr": float(r.snr), "n_left": int(r.n_left), "n_right": int(r.n_right),
             "tau": float(r.tau_ns)}
            for r in g.itertuples()
        ]
        kept = {}
        for name in tot:
            items_sorted = sorted(rows, key=KEYS[name])
            kv, _ = probe.replay(items_sorted, 0.02, dt, probe.F_C_4TU, "current")
            kept[name] = kv
            taus[name].extend(s["tau"] for s in kv)

        def near(p, q):
            return abs(p["trace"] - q["trace"]) <= 3 and abs(p["t_ns"] - q["t_ns"]) <= 1.0

        a = kept["cur"]
        for name in tot:
            b = kept[name]
            tot[name]["n_kept"] += len(b)
            tot[name]["lines_changed"] += int(
                sorted((s["trace"], s["t_ns"]) for s in a) != sorted((s["trace"], s["t_ns"]) for s in b)
            )
            tot[name]["new_far"] += sum(not any(near(q, p) for p in a) for q in b)
            tot[name]["lost_far"] += sum(not any(near(p, q) for q in b) for p in a)
    for name in tot:
        s = pd.Series(taus[name])
        tot[name]["median_tau_kept"] = round(float(s.median()), 3)
        tot[name]["p90_tau_kept"] = round(float(s.quantile(0.9)), 3)
    return tot


_ORIG_RS = gen._rank_and_suppress


def run_combo(rg, inputs, act, path, snap, key_name, vertex=False):
    """Generator with K7-a1 start snap (``snap``), K7-c1 (``vertex``) and a rank key."""
    key = KEYS[key_name]

    def rs(accepted, dx):
        if key_name == "b3":
            E = gen_state["E"]
            for s in accepted:
                s["E"] = float(E[int(round(s["sample"])), s["trace"]])
        accepted.sort(key=key)
        return _ORIG_RS(accepted, dx)

    probe._STATE.update(snap=snap, vertex=vertex, t_z=inputs.zero.t_zero_ns)
    gen.crest_apex = probe.patched_crest
    gen._rank_and_suppress = rs
    try:
        return gen.apex_candidates_for_radargram(rg, inputs, act=act, path=path)
    finally:
        gen.crest_apex = probe._ORIG_CREST
        gen._rank_and_suppress = _ORIG_RS


gen_state = {"E": None}
COMBOS = {
    "cur": (False, "cur", False),
    "a1": (True, "cur", False),
    "a1b2": (True, "b2", False),
    "a1b3": (True, "b3", False),
    "a1b4": (True, "b4", False),
    "a1b2c1": (True, "b2", True),
}


def s1g_combo_part():
    out = {}
    for model in ("S1G_C1", "S1G_D060", "S1G_D100", "S1G_BGL"):
        rg = s1g.load_line(model, probe.S1G_ROOT)
        meta = json.loads((probe.S1G_ROOT / f"{model}.json").read_text(encoding="utf-8"))
        inputs = prepare_depth_inputs(rg, DepthConfig())
        _, _, E0 = gen.apex_candidates_for_radargram(rg, inputs, act="S1G", path=model)
        gen_state["E"] = E0
        res = {}
        for name, (snap, key_name, vertex) in COMBOS.items():
            line, kept, _ = run_combo(rg, inputs, "S1G", model, snap, key_name, vertex)
            per = []
            for p in meta.get("pipes") or []:
                ja, ta = p["apex_trace_index"], p["apex"]["envelope_peak_ns"]
                per.append(
                    sum(abs(s["trace"] - ja) <= 3 and abs(s["t_ns"] - ta) <= 1.0 for s in kept)
                )
            res[name] = {"kept_in_window_per_pipe": per, "n_kept": len(kept),
                         "kept_t_ns": [s["t_ns"] for s in kept]}
        out[model] = res
    return out


def fourtu_combo_part():
    """a1b2 / a1b4 against current on the 4TU lines (already seen, exploration)."""
    lines = pd.read_csv(probe.LINES, dtype=probe.ID_COLUMNS)
    limit = int(os.environ.get("K7_PROBE_LINES", "0"))
    if limit:
        lines = lines.head(limit)
    names = ["cur", "a1b2", "a1b4"]
    tot = {n: {"n_kept": 0, "lines_changed": 0, "new_far": 0, "lost_far": 0} for n in names}
    n_lines = 0
    for r in lines.itertuples():
        rg = gen.load_fourtu_radargram(gen.BASE, r.activity_id, r.path)
        inputs = gen.prepare_depth_inputs(rg)
        if inputs.zero.t_zero_ns is None:
            continue
        n_lines += 1
        kept = {}
        for name in names:
            snap, key_name, vertex = COMBOS.get(name, (True, name[2:], False))
            _, kv, _ = run_combo(rg, inputs, r.activity_id, r.path, snap, key_name, vertex)
            kept[name] = [(s["trace"], s["t_ns"]) for s in kv]

        def near(p, q):
            return abs(p[0] - q[0]) <= 3 and abs(p[1] - q[1]) <= 1.0

        for name in names:
            a, b = kept["cur"], kept[name]
            tot[name]["n_kept"] += len(b)
            tot[name]["lines_changed"] += int(sorted(a) != sorted(b))
            tot[name]["new_far"] += sum(not any(near(q, p) for p in a) for q in b)
            tot[name]["lost_far"] += sum(not any(near(p, q) for q in b) for p in a)
    return {"n_lines": n_lines, "variants": tot}


def fourtu_c1_loss_part():
    """Which 4TU kept items does a1c1 lose / add (already seen, exploration)."""
    lines = pd.read_csv(probe.LINES, dtype=probe.ID_COLUMNS)
    lost, new, all_a, all_b = [], [], [], []
    for r in lines.itertuples():
        rg = gen.load_fourtu_radargram(gen.BASE, r.activity_id, r.path)
        inputs = gen.prepare_depth_inputs(rg)
        if inputs.zero.t_zero_ns is None:
            continue
        _, a, _ = run_combo(rg, inputs, r.activity_id, r.path, False, "cur", False)
        _, b, _ = run_combo(rg, inputs, r.activity_id, r.path, True, "cur", True)

        def near(p, q):
            return abs(p["trace"] - q["trace"]) <= 3 and abs(p["t_ns"] - q["t_ns"]) <= 1.0

        all_a += a
        all_b += b
        lost += [p for p in a if not any(near(p, q) for q in b)]
        new += [q for q in b if not any(near(q, p) for p in a)]

    def summ(rows, ref):
        if not rows:
            return {"n": 0}
        f = pd.DataFrame(rows)
        mw = f[["n_left", "n_right"]].min(axis=1)
        g = pd.DataFrame(ref)
        mw_ref = g[["n_left", "n_right"]].min(axis=1)
        return {
            "n": len(f),
            "frac_min_wing_ge40": round(float((mw >= 40).mean()), 3),
            "ref_frac_min_wing_ge40": round(float((mw_ref >= 40).mean()), 3),
            "median_v_straight_qc": round(float(f["v_straight_qc"].median()), 4),
            "ref_median_v_straight_qc": round(float(g["v_straight_qc"].median()), 4),
            "median_tau_ns": round(float(f["tau_ns"].median()), 3),
            "ref_median_tau_ns": round(float(g["tau_ns"].median()), 3),
        }

    return {"lost_far_vs_all_cur_kept": summ(lost, all_a),
            "new_far_vs_all_a1c1_kept": summ(new, all_b)}


def main():
    if os.environ.get("K7_C1_LOSS", "0") == "1":
        out = json.loads(OUT.read_text(encoding="utf-8"))
        out["fourtu_c1_loss"] = fourtu_c1_loss_part()
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"out": str(OUT), "ok": True}, ensure_ascii=True))
        return
    out = {"note": "already-seen data, development evidence only",
           "s1g": s1g_part(), "fourtu_541_lines": fourtu_part(),
           "s1g_combo": s1g_combo_part()}
    if os.environ.get("K7_RANK_4TU", "0") == "1":
        out["fourtu_combo"] = fourtu_combo_part()
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "ok": True}, ensure_ascii=True))


if __name__ == "__main__":
    main()
