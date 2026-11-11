"""M2-10 (b) pure-function reproduction of the selector floor / time-zero coupling.

Synthetic, no 4TU data. Builds background-removed sections S directly
(band-limited Gaussian noise + a laterally zero-mean strong shallow event + one
deep hyperbola, amplitude A_d d / r along the wings), runs the detection step of ``m210b_floor_rules`` for every rule
with two mute positions, and the full candidate path (crest move + acceptance of
``fourtu_apex_candidate_select``) for the apex and for noise-only sections.

All values below are ASSUMPTIONS of the prototype (4TU-like scales, not fitted):
dt 0.098 ns, dx 0.02 m, f_c 500 MHz, 512 samples x 301 traces, eps_r 9,
noise sigma_n = 1 (after band-limiting), mute positions t_z 3.03 / 5.38 ns
(06.6 Path10 v2.1 / v2.0 zero, spec_eval 9-7 E), strong event at 5.38 ns.

Cases (document section 4-1, expected values written there):
  P1 coupling grid   A_s in {50, 100, 200, 400}, A_d in {3, 4, 5, 6, 8, 12, 16}, d 1.3 m
  P2 noise only      10 seeds: detection peaks and accepted false candidates per rule
  P3 sensitivity     no strong event, A_d grid: apex detected per rule
  P4 residual A      strong event 1.5 periods below the new mute (A_skip1 still coupled)
  P5 floor shift     floor ratio new/old mute per rule (MAD bound for B_noise)
  P6 noise free      numerical guard of the noise floors
  P7 non-stationary  noise sigma 0.3 -> 1.5 over the record (gain-like), false peaks per rule

Output: docs/processing/m210b_floor_proto.json (+ printed summary).
Seeds: noise seed = 20261111 + i (i = case-local index), fixed.
Run: conda run -n gpr-dev python -I analysis/processing/m210b_floor_proto.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))

import fourtu_apex_candidate_select as gen  # noqa: E402
import m210b_floor_rules as fr  # noqa: E402

DT, DX, F_C = 0.098, 0.02, 500e6
NS, NT = 512, 301
EPS_R = 9.0
V = 0.299792458 / math.sqrt(EPS_R)
T_NEW, T_OLD, T_STRONG = 3.03, 5.38, 5.38
D_PIPE = 1.3
X0 = (NT // 2) * DX
EDGE = int(math.ceil(gen.EDGE_M / DX))
SEED0 = 20261111
LATERAL_PERIOD_M = 1.0  # strong event amplitude ~ sin(2 pi x / 1 m): zero lateral mean
HIT_TRACES = 3
HIT_PERIODS = 0.5  # apex hit: peak within +-2 traces and +-0.5 period of the true apex


def ricker(t_ns: np.ndarray) -> np.ndarray:
    a = (np.pi * F_C * 1e-9 * t_ns) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, (NS, NT))
    kern = ricker((np.arange(-40, 41)) * DT)
    out = fftconvolve(w, kern[:, None], mode="same", axes=0)
    return out / out.std()


def section(seed: int | None, a_strong: float, a_deep: float, t_strong: float = T_STRONG, d: float = D_PIPE) -> np.ndarray:
    t = np.arange(NS)[:, None] * DT
    x = np.arange(NT) * DX
    s = np.zeros((NS, NT)) if seed is None else noise(seed)
    if a_strong:
        s += a_strong * np.sin(2 * np.pi * x / LATERAL_PERIOD_M)[None, :] * ricker(t - t_strong)
    if a_deep:
        # zero = surface reflection at T_NEW (pipe times relative to the v2.1 zero)
        r = np.sqrt(d**2 + (x - X0) ** 2)
        tp = T_NEW + 2.0 * r / V
        # amplitude falls along the wings (d / r, spreading): the apex is the
        # strict crest maximum, as in real data (a flat crest makes plateau ties)
        s += a_deep * (d / r)[None, :] * ricker(t - tp[None, :])
    return s


def envelope(s: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(s, axis=0))


def apex_row(d: float = D_PIPE) -> float:
    return (T_NEW + 2.0 * d / V) / DT


def run_rule(E, S, t_z, rule, cap=np.inf):
    k_lo, k_hi = fr.search_rows(t_z, F_C, DT, NS)
    pk, dg, level, lam = fr.detect(E, S, k_lo, k_hi, F_C, DT, EDGE, rule, cap)
    return pk, dg, k_lo, k_hi, level


def crest_row(j: int, d: float = D_PIPE) -> float:
    return (T_NEW + 2.0 * math.hypot(d, j * DX - X0) / V) / DT


def apex_hit(pk, k_a: float, j_a: int, E=None, t_z=None, k_lo=None) -> tuple[bool, float]:
    """Peak on the target crest within the crest-move range (+-0.5 m, +-0.5 period).

    With ``E`` given, the hit also requires that the peak becomes an accepted
    candidate (crest move + acceptance of selector v1) within +-HIT_TRACES of the
    apex and +-0.5 period (the selector's own path from a peak to a candidate).
    """
    tol_k = HIT_PERIODS * 1e9 / F_C / DT
    reach = int(round(0.5 / DX))
    on = [p for p in pk if abs(p[1] - j_a) <= reach and abs(p[0] - crest_row(p[1])) <= tol_k]
    if E is None:
        return bool(on), max((p[2] for p in on), default=float("nan"))
    acc = accepted(E, on, t_z, k_lo)
    ok = [a for a in acc if abs(a[0] - j_a) <= HIT_TRACES and abs(a[1] - k_a) <= tol_k]
    return bool(ok), max((p[2] for p in on), default=float("nan"))


def accepted(E, pk, t_z, k_lo):
    """Run crest move + acceptance (selector v1) on peaks; return accepted (j, k)."""
    out = []
    for k_int, j0 in [(p[0], p[1]) for p in pk]:
        k0, _ = gen.parabolic_peak(E[:, j0], k_int)
        j, k = gen.crest_apex(E, j0, k0, DT, DX)
        if not EDGE <= j < NT - EDGE or k < k_lo:
            continue
        v, *_ = gen.accept(E, j, k, DT, DX, t_z)
        if v is not None:
            out.append((j, k))
    return out


def p1_coupling() -> list[dict]:
    rows = []
    k_a, j_a = apex_row(), NT // 2
    for i, a_s in enumerate((50, 100, 200, 400)):
        for a_d in (3, 4, 5, 6, 8, 12, 16):
            S = section(SEED0 + i, a_s, a_d)
            E = envelope(S)
            sig = fr.noise_sigma(S[fr.search_rows(T_NEW, F_C, DT, NS)[0]: NS - fr.RECORD_END_TRIM])
            e_a = float(E[int(round(k_a)), j_a])
            r = {"A_s": a_s, "A_d": a_d, "seed": SEED0 + i, "E_apex_over_5sigma": round(e_a / (fr.K_NOISE * sig), 3)}
            for rule in fr.RULES:
                for tz_name, tz in (("new", T_NEW), ("old", T_OLD)):
                    pk, dg, k_lo, _, level = run_rule(E, S, tz, rule, fr.NUM_PEAKS)
                    hit, dval = apex_hit(pk, k_a, j_a)
                    r[f"{rule}_{tz_name}_peak"] = hit
                    r[f"{rule}_{tz_name}_hit"] = apex_hit(pk, k_a, j_a, E, tz, k_lo)[0]
                    r[f"{rule}_{tz_name}_D"] = None if math.isnan(dval) else round(dval, 2)
                    r[f"{rule}_{tz_name}_F"] = round(dg["F"], 3)
                    r[f"{rule}_{tz_name}_floor_at_apex"] = bool(level[int(round(k_a)) - k_lo] < dg["F"])
            rows.append(r)
    return rows


def p2_noise_only(n_seeds: int = 10) -> dict:
    out = {r: {"peaks_nocap": [], "peaks_cap300": [], "accepted_cap300": []} for r in fr.RULES}
    for i in range(n_seeds):
        S = section(SEED0 + 100 + i, 0.0, 0.0)
        E = envelope(S)
        for rule in fr.RULES:
            pk_inf, *_ = run_rule(E, S, T_NEW, rule, np.inf)
            pk, _, k_lo, _, _ = run_rule(E, S, T_NEW, rule, fr.NUM_PEAKS)
            out[rule]["peaks_nocap"].append(len(pk_inf))
            out[rule]["peaks_cap300"].append(len(pk))
            out[rule]["accepted_cap300"].append(len(accepted(E, pk, T_NEW, k_lo)))
    return {
        r: {k: {"mean": round(float(np.mean(v)), 2), "max": int(np.max(v)), "values": v} for k, v in d.items()}
        for r, d in out.items()
    }


def p3_sensitivity(n_seeds: int = 10) -> list[dict]:
    """No strong event: apex peak / candidate rate per rule versus A_d (10 seeds each)."""
    rows = []
    k_a, j_a = apex_row(), NT // 2
    for i, a_d in enumerate((2.5, 3, 4, 5, 6, 8, 10, 12)):
        r = {"A_d": a_d, "n_seeds": n_seeds, "E_apex_over_5sigma": []}
        for rule in fr.RULES:
            r[f"{rule}_peak"] = 0
            r[f"{rule}_hit"] = 0
        for s in range(n_seeds):
            S = section(SEED0 + 1000 + 100 * i + s, 0.0, a_d)
            E = envelope(S)
            k_lo = fr.search_rows(T_NEW, F_C, DT, NS)[0]
            sig = fr.noise_sigma(S[k_lo: NS - fr.RECORD_END_TRIM])
            r["E_apex_over_5sigma"].append(round(float(E[int(round(k_a)), j_a]) / (fr.K_NOISE * sig), 3))
            for rule in fr.RULES:
                pk, _, k_lo, _, _ = run_rule(E, S, T_NEW, rule, fr.NUM_PEAKS)
                r[f"{rule}_peak"] += apex_hit(pk, k_a, j_a)[0]
                r[f"{rule}_hit"] += apex_hit(pk, k_a, j_a, E, T_NEW, k_lo)[0]
        r["E_apex_over_5sigma_median"] = float(np.median(r.pop("E_apex_over_5sigma")))
        rows.append(r)
    return rows


def p4_residual_A() -> list[dict]:
    rows = []
    k_lo_new = fr.search_rows(T_NEW, F_C, DT, NS)[0]
    t_strong = k_lo_new * DT + 1.5 * 1e9 / F_C  # 1.5 periods below the new mute
    t_old = t_strong + 0.5  # v2.0-like zero sitting just after the strong event
    k_a, j_a = apex_row(), NT // 2
    for i, a_s in enumerate((100, 200, 400)):
        S = section(SEED0 + 300 + i, a_s, 12.0, t_strong=t_strong)
        E = envelope(S)
        r = {"A_s": a_s, "A_d": 12.0, "t_strong_ns": round(t_strong, 3), "t_old_ns": round(t_old, 3)}
        for rule in fr.RULES:
            for tz_name, tz in (("new", T_NEW), ("old", t_old)):
                pk, dg, k_lo, *_ = run_rule(E, S, tz, rule, fr.NUM_PEAKS)
                r[f"{rule}_{tz_name}_peak"] = apex_hit(pk, k_a, j_a)[0]
                r[f"{rule}_{tz_name}_hit"] = apex_hit(pk, k_a, j_a, E, tz, k_lo)[0]
                r[f"{rule}_{tz_name}_F"] = round(dg["F"], 3)
        rows.append(r)
    return rows


def p5_floor_shift() -> dict:
    out = {}
    for i, a_s in enumerate((50, 100, 200, 400)):
        S = section(SEED0 + i, a_s, 8.0)
        E = envelope(S)
        for rule in fr.RULES:
            f_new = run_rule(E, S, T_NEW, rule)[1]["F"]
            f_old = run_rule(E, S, T_OLD, rule)[1]["F"]
            out.setdefault(rule, []).append(round(f_new / f_old, 4))
    return out


def p7_nonstationary(n_seeds: int = 10, a_strong: float = 0.0) -> dict:
    """Noise sigma rising linearly with time 0.3 -> 1.5 (field gain, ASSUMPTION); optional strong event."""
    ramp = np.linspace(0.3, 1.5, NS)[:, None]
    out = {r: {"peaks_cap300": [], "accepted_cap300": []} for r in fr.RULES}
    x = np.arange(NT) * DX
    t = np.arange(NS)[:, None] * DT
    for i in range(n_seeds):
        S = noise(SEED0 + 700 + i) * ramp
        if a_strong:
            S = S + a_strong * np.sin(2 * np.pi * x / LATERAL_PERIOD_M)[None, :] * ricker(t - T_STRONG)
        E = envelope(S)
        for rule in fr.RULES:
            pk, _, k_lo, _, _ = run_rule(E, S, T_NEW, rule, fr.NUM_PEAKS)
            out[rule]["peaks_cap300"].append(len(pk))
            out[rule]["accepted_cap300"].append(len(accepted(E, pk, T_NEW, k_lo)))
    return {
        r: {k: {"mean": round(float(np.mean(v)), 2), "max": int(np.max(v))} for k, v in d.items()}
        for r, d in out.items()
    }


def p7b_stationary_strong(n_seeds: int = 10) -> dict:
    """Stationary noise + strong event A_s 200 (4TU-like shallow clutter), no pipe."""
    out = {r: {"peaks_cap300": [], "accepted_cap300": []} for r in fr.RULES}
    for i in range(n_seeds):
        S = section(SEED0 + 800 + i, 200.0, 0.0)
        E = envelope(S)
        for rule in fr.RULES:
            pk, _, k_lo, _, _ = run_rule(E, S, T_NEW, rule, fr.NUM_PEAKS)
            out[rule]["peaks_cap300"].append(len(pk))
            out[rule]["accepted_cap300"].append(len(accepted(E, pk, T_NEW, k_lo)))
    return {
        r: {k: {"mean": round(float(np.mean(v)), 2), "max": int(np.max(v))} for k, v in d.items()}
        for r, d in out.items()
    }


def p6_noise_free() -> dict:
    S = section(None, 200.0, 8.0)
    E = envelope(S)
    res = {}
    for rule in fr.RULES:
        pk, dg, k_lo, *_ = run_rule(E, S, T_NEW, rule, np.inf)
        res[rule] = {"F": dg["F"], "n_peaks_nocap": len(pk), "apex_peak": apex_hit(pk, apex_row(), NT // 2)[0],
                     "apex_hit": apex_hit(pk, apex_row(), NT // 2, E, T_NEW, k_lo)[0]}
    return res


def main() -> None:
    p1 = p1_coupling()
    summary = {
        "note": "pure-function synthetic prototype (no 4TU); values are expected values for tests",
        "assumptions": {
            "dt_ns": DT, "dx_m": DX, "f_c_hz": F_C, "n_samples": NS, "n_traces": NT, "eps_r": EPS_R,
            "t_zero_new_ns": T_NEW, "t_zero_old_ns": T_OLD, "t_strong_ns": T_STRONG, "d_pipe_m": D_PIPE,
            "sigma_n": 1.0, "lateral_period_m": LATERAL_PERIOD_M, "seed0": SEED0,
            "hit": f"+-{HIT_TRACES} traces, +-{HIT_PERIODS} period",
        },
        "P1_coupling": p1,
        "P1_counts": {
            f"{rule}_{tz}": int(sum(r[f"{rule}_{tz}_hit"] for r in p1)) for rule in fr.RULES for tz in ("new", "old")
        },
        "P1_counts_peak_stage": {
            f"{rule}_{tz}": int(sum(r[f"{rule}_{tz}_peak"] for r in p1)) for rule in fr.RULES for tz in ("new", "old")
        },
        "P1_n_cases": len(p1),
        "P1_flip_old_hit_new_miss": {
            rule: int(sum(r[f"{rule}_old_hit"] and not r[f"{rule}_new_hit"] for r in p1)) for rule in fr.RULES
        },
        "P1_flip_any": {rule: int(sum(r[f"{rule}_old_hit"] != r[f"{rule}_new_hit"] for r in p1)) for rule in fr.RULES},
        "P2_noise_only": p2_noise_only(),
        "P3_sensitivity": p3_sensitivity(),
        "P4_residual_A": p4_residual_A(),
        "P5_floor_ratio_new_over_old": p5_floor_shift(),
        "P6_noise_free": p6_noise_free(),
        "P7_nonstationary_noise": p7_nonstationary(),
        "P7b_nonstationary_with_strong_event_200": p7_nonstationary(a_strong=200.0),
        "P2b_stationary_with_strong_event_200": p7b_stationary_strong(),
    }
    out = ROOT / "docs" / "processing" / "m210b_floor_proto.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("P1_coupling",)}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
