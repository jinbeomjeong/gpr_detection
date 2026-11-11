"""M2-10 K1 S1g verdict support (spec_eval_4tu_v1 v1.9 section 12-4, sealed b3e68fa).

Reads the harness output (``data/processing/m210_s1g``) and recomputes the
numbers of G0-G7. It also traces *why* primaries are missing (diagnostic only,
does not change the sealed rules):

* generator stage per primary: peaks near the primary apex, crest move,
  acceptance reason, rank and kept / NMS / per-line cap;
* depth stage: ``estimate_depth`` with the hint on the primary's own apex;
* sampling: the same harness path on the lines decimated by 8 (dt ~0.094 ns,
  close to 4TU 0.097-0.098 ns).

No 4TU data. Output ``docs/processing/m210_k1_s1g_judgment.json``.

Run (one core, BelowNormal):
    conda run -n gpr-dev python -I analysis/processing/m210_k1_s1g_judge.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import eval_depth_4tu as ev  # noqa: E402
import fourtu_apex_candidate_select as gen  # noqa: E402
import m210_s1g_pipeline as har  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import estimate_depth, prepare_depth_inputs  # noqa: E402
from gpr_detection.depth.picks import ApexHint  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

OUT = ROOT / "docs" / "processing" / "m210_k1_s1g_judgment.json"
HARNESS = ROOT / "data" / "processing" / "m210_s1g"
DATA = ROOT / "data" / "synthetic_gprmax_s1g"
MODELS = ["S1G_C1", "S1G_D060", "S1G_D100", "S1G_BGL"]
D_TOP = 0.60
W = ev.CLUSTER_WIDTH_M


def load_events():
    js = json.loads((ROOT / "analysis" / "synthetic" / "s1g" / "models_s1g.json").read_text(encoding="utf-8"))
    return js["expected_events"]


def primaries(events, model):
    return [e for e in events.get(model, []) if e["kind"] == "primary"]


def g_table(cands: pd.DataFrame, events) -> dict:
    out = {}
    for model in MODELS:
        c = cands[cands["model_id"] == model]
        qp = c[(c["status"] == "ok") & (~c["qc_excluded"].astype(str).eq("True"))]
        prim = qp[qp["event_kind"] == "primary"]
        kinds = qp["event_kind"].value_counts().to_dict()
        clusters = {}
        for _, r in qp.iterrows():
            cid = int(r["cluster_id"])
            cl = clusters.setdefault(cid, {"depth_m": float(r["cluster_depth_m"]), "mu": int(r["cluster_mu"]), "members": []})
            cl["members"].append({"apex_x_m": float(r["apex_x_m"]), "t_ns": float(r["apex_t_ns"]),
                                  "depth": float(r["depth_top_m"]), "kind": r["event_kind"]})
        pipe_band = [cid for cid, cl in clusters.items() if abs(cl["depth_m"] - (D_TOP + 0.017)) <= W]
        out[model] = {
            "n_candidates": int(len(c)),
            "n_qc_passing": int(len(qp)),
            "kinds_qc_passing": {str(k): int(v) for k, v in kinds.items()},
            "primaries_expected": [{"x_m": e["x_m"], "t_ns": e["t_ns"], "pipe_index": e.get("pipe_index")} for e in primaries(events, model)],
            "primary_candidates": [{"apex_x_m": float(r["apex_x_m"]), "apex_t_ns": float(r["apex_t_ns"]),
                                    "depth_top_m": float(r["depth_top_m"]), "err_m": round(float(r["depth_top_m"]) - D_TOP, 4),
                                    "pipe_index": r["event_pipe_index"]} for _, r in prim.iterrows()],
            "clusters": clusters,
            "clusters_within_W_of_primary_depth_0617": pipe_band,
            "n_m1": int(kinds.get("multiple_surface", 0)),
            "n_m2": int(kinds.get("multiple_pipe", 0)),
            "n_m3": int(kinds.get("multiple_other", 0)),
            "n_u": int(kinds.get("unexplained", 0)),
            "qc_passing_in_050_070": int(((qp["depth_top_m"] >= 0.50) & (qp["depth_top_m"] <= 0.70)).sum()),
        }
    return out


def generator_trace(rg, inputs, prim_events):
    """Per-peak stage near each primary (same rules as apex_candidates_for_radargram)."""
    dt, dx = rg.dt_ns, rg.dx_m
    zero = inputs.zero
    E = pick_signal(inputs.branch.background_removed, "envelope")
    k_lo = math.ceil((zero.t_zero_ns + 1e9 / rg.f_c_hz) / dt) + 2
    k_hi = E.shape[0] - 16
    sub = E[k_lo:k_hi]
    level = uniform_filter1d(np.median(sub, axis=1), size=max(3, int(1e9 / rg.f_c_hz / dt)))
    level = np.maximum(level, 0.05 * float(level.max()))
    detect = sub / level[:, None]
    peaks = peak_local_max(detect, min_distance=8, threshold_abs=gen.SNR_MIN,
                           exclude_border=False, num_peaks=gen.MAX_CANDIDATES)
    n_raw = len(peaks)
    edge = int(math.ceil(gen.EDGE_M / dx))
    peaks = [(int(k) + k_lo, int(j)) for k, j in peaks if edge <= j < rg.n_traces - edge]
    x0 = rg.metadata["trace_x_m_first"]
    items = []
    for k_int, j0 in peaks:
        k0, _ = parabolic_peak(E[:, j0], k_int)
        j, k = gen.crest_apex(E, j0, k0, dt, dx)
        rec = {"peak_trace": j0, "peak_t_ns": round(k0 * dt, 3), "trace": j, "t_ns": round(k * dt, 3),
               "snr": round(float(detect[min(int(round(k)), k_hi - 1) - k_lo, j]), 2) if k >= k_lo else None}
        if not edge <= j < rg.n_traces - edge or k < k_lo:
            rec["reason"] = "apex_at_edge_or_mute"
        else:
            v, nl, nr, why = gen.accept(E, j, k, dt, dx, zero.t_zero_ns)
            rec.update({"n_left": nl, "n_right": nr, "reason": why if v is not None or why else ""})
            if v is not None:
                rec["reason"] = "accepted"
                rec["v"] = round(v, 4)
        items.append(rec)
    acc = [r for r in items if r["reason"] == "accepted"]
    acc.sort(key=lambda s: (-min(s["n_left"], s["n_right"]), -s["snr"]))
    kept = []
    for rank, s in enumerate(acc):
        if len(kept) >= gen.MAX_APEX_CANDIDATES_PER_LINE:
            s["fate"] = "cap"
            continue
        clash = [o for o in kept if not (abs(s["trace"] - o["trace"]) * dx > gen.NMS_M or abs(s["t_ns"] - o["t_ns"]) > gen.NMS_NS)]
        s["rank"] = rank
        if clash:
            s["fate"] = f"nms_by_trace{clash[0]['trace']}_t{clash[0]['t_ns']}"
        else:
            s["fate"] = "kept"
            kept.append(s)
    near = {}
    for e in prim_events:
        jp = round((e["x_m"] - x0) / dx)
        near[f"pipe{e.get('pipe_index')}_trace{jp}"] = [
            r for r in items if abs(r["peak_trace"] - jp) <= 8 and abs(r["peak_t_ns"] - e["t_ns"]) <= 3.0
        ]
    return {
        "n_peaks_raw": n_raw, "n_peaks_edge_ok": len(peaks), "num_peaks_cap": gen.MAX_CANDIDATES,
        "n_accepted": len(acc),
        "accepted_ranked": [{k: s.get(k) for k in ("trace", "t_ns", "n_left", "n_right", "snr", "rank", "fate")} for s in acc],
        "near_primary_peaks": near,
    }


def depth_with_true_hint(rg, inputs, prim_events):
    x0 = rg.metadata["trace_x_m_first"]
    out = []
    zero = inputs.zero
    for e in prim_events:
        trace = round((e["x_m"] - x0) / rg.dx_m)
        sample = e["t_ns"] / rg.dt_ns
        tau = e["t_ns"] - zero.t_zero_ns
        box = ev.candidate_box(rg, trace, sample, tau)
        res = estimate_depth(rg, box, DepthConfig(), inputs=inputs, apex_hint=ApexHint(trace=trace, sample=sample))
        out.append({"pipe_index": e.get("pipe_index"), "hint_trace": trace, "hint_t_ns": e["t_ns"],
                    "status": str(res.status), "depth_top_m": None if res.depth_top_m is None else round(res.depth_top_m, 4),
                    "apex_trace": res.apex_trace, "flags": list(res.flags)})
    return out


def decimated_run(model, events, factor=8):
    rg = har.load_line(model, DATA)
    data = resample_poly(rg.data.astype(np.float64), up=1, down=factor, axis=0).astype(np.float32)
    rg2 = dataclasses.replace(rg, data=data, dt_ns=rg.dt_ns * factor)
    rows, info = har.run_model(model, rg2, har.parse_events(events.get(model, [])))
    keep = ["candidate_trace", "apex_x_m", "apex_t_ns", "depth_top_m", "status", "qc_verdict", "qc_reason",
            "qc_excluded", "event_kind", "cluster_id", "cluster_depth_m", "cluster_mu"]
    return {"dt_ns": rg2.dt_ns, "zero_rule": info["zero_rule"], "t_zero_ns": info["t_zero_ns"],
            "k1": info["k1"], "rows": [{k: r.get(k) for k in keep} for r in rows]}


def main():
    events = load_events()
    cands = pd.read_csv(HARNESS / "candidates.csv", dtype={"model_id": str})
    out = {"seal_commit": "b3e68fa", "spec_blob": "0f368eec79b6dfc8fdb8430484a1f1618160ec6c",
           "g_table": g_table(cands, events), "generator": {}, "depth_true_hint": {}, "decimated_x8": {}}
    for model in ("S1G_C1", "S1G_D060", "S1G_D100"):
        rg = har.load_line(model, DATA)
        inputs = prepare_depth_inputs(rg, DepthConfig())
        prim = primaries(events, model)
        out["generator"][model] = generator_trace(rg, inputs, prim)
        out["depth_true_hint"][model] = depth_with_true_hint(rg, inputs, prim)
    for model in MODELS:
        out["decimated_x8"][model] = decimated_run(model, events)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    brief = {
        m: {"prim": out["g_table"][m]["primary_candidates"], "kinds": out["g_table"][m]["kinds_qc_passing"],
            "band": out["g_table"][m]["clusters_within_W_of_primary_depth_0617"]} for m in MODELS
    }
    print(json.dumps(brief, default=str))
    for m, g in out["generator"].items():
        print(m, "accepted", g["n_accepted"], json.dumps(g["accepted_ranked"][:10]))
        for key, lst in g["near_primary_peaks"].items():
            print("  ", key, json.dumps(lst))
    print("true-hint", json.dumps(out["depth_true_hint"], default=str))
    for m, d in out["decimated_x8"].items():
        print("x8", m, d["dt_ns"], d["zero_rule"], json.dumps(d["k1"]), json.dumps(
            [(r["candidate_trace"], r["apex_t_ns"], r["depth_top_m"], r["status"], r["qc_verdict"], r["event_kind"], r["cluster_mu"]) for r in d["rows"]]))


if __name__ == "__main__":
    main()
