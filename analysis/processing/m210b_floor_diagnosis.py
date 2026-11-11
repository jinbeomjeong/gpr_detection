"""M2-10 (b) selector detection floor vs time zero on 4TU (ALREADY-SEEN DATA, EXPLORATION).

4TU results were seen in the M2-9 re-evaluation. Everything here is a cause
diagnosis; no threshold is fitted and no rule is chosen from these numbers
(docs/processing/m210b_selector_floor.md, M2-10 principle).

Parts (one pass over the radargrams):

1. Label-free, all 759 lines: where max l(k) sits relative to the mute, the
   fraction of searched rows held by the floor, sigma_n, the floor of every rule
   and the detection-peak count per rule (cap 300 and no cap). The ``current``
   cap-300 count must equal ``n_peaks`` of ``fourtu_apex_candidate_lines_v1.csv``
   (reproduction check).
2. Label-free, the 50 two-event lines: detection peaks below the v2.0 mute + one
   period with the v2.0 zero (git ``bbf484b~1`` lines csv) and with the v2.1 zero,
   per rule. A rule is zero-decoupled when the two sets agree.
3. Label-free, kept auto candidates: envelope at the candidate / (K_NOISE sigma_n)
   versus the depth-path flag ``apex_hint_unmatched``.
4. GT (already seen): the 16 truths of ``m210d_truth_classes.csv`` with mechanism
   ``i-gen0 no peak``. For each line of the activity and each rule x cap the
   detection peaks within +-0.30 m (nominal depth v_meta tau / 2) and whether a
   peak there passes ``crest_apex`` + ``accept``. Classes fixed before the run
   (document section 3-1): crest_moved_out / cap300 / floor (E >= 5 sigma_n) /
   floor_noise_level / no_energy.

Outputs:
* data/processing/m210b_lines.csv            (part 1, one row per line, git-ignored)
* data/processing/m210b_two_event.csv        (part 2, one row per line x rule x cap)
* data/processing/m210b_truth_lines.csv      (part 4, one row per truth x line)
* docs/processing/m210b_truth_no_peak.csv    (part 4, one row per truth)
* docs/processing/m210b_floor_diagnosis.json (summary)

Run (one E core, BelowNormal, BLAS threads 1):
    conda run -n gpr-dev python -I analysis/processing/m210b_floor_diagnosis.py
"""

import argparse
import io
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "processing"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_depth_4tu as ev  # noqa: E402  (read-only import, PASS_LIMIT_M)
import fourtu_apex_candidate_select as gen  # noqa: E402
import m210b_floor_rules as fr  # noqa: E402

from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402
from gpr_detection.io.fourtu import load_fourtu_radargram  # noqa: E402
from gpr_detection.io.radargram import velocity_from_eps  # noqa: E402
from gpr_detection.preprocess.signals import parabolic_peak, pick_signal  # noqa: E402

BASE = gen.BASE
DOCS = ROOT / "docs" / "processing"
DATA = ROOT / "data" / "processing"
ID = {"activity_id": str, "path": str, "line_id": str}
LIMIT = ev.PASS_LIMIT_M + ev.PASS_TOL_M
GRID = np.round(np.arange(0.15, 2.505, 0.01), 3)  # m210d chance-coverage grid
OLD_ZERO_REV = "bbf484b~1"  # last commit with the v2.0 zero in the lines csv
MATCH_TRACES, MATCH_SAMPLES = 3, 5  # spec_eval 9-7 (b) reproduction match
CAPS = {"cap300": fr.NUM_PEAKS, "nocap": np.inf}
NO_PEAK = "i-gen0 no peak (no detectable energy)"


def read_csv_git(rev: str, rel: str) -> pd.DataFrame:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return pd.read_csv(io.StringIO(out), dtype=ID)


def match_count(a: list, b: list) -> int:
    """Items of ``a`` with a partner in ``b`` within +-3 traces and +-5 samples."""
    if not a:
        return 0
    if not b:
        return 0
    bk = np.array([p[0] for p in b])
    bj = np.array([p[1] for p in b])
    return int(
        sum(
            np.any((np.abs(bk - k) <= MATCH_SAMPLES) & (np.abs(bj - j) <= MATCH_TRACES))
            for k, j, _ in a
        )
    )


def coverage(depths: list[float]) -> float:
    if not depths:
        return 0.0
    d = np.asarray(depths)
    return float(np.mean(np.any(np.abs(GRID[:, None] - d[None, :]) <= LIMIT, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", default="", help="restrict to these activities (smoke)")
    args = parser.parse_args()
    started = time.perf_counter()
    index = pd.read_csv(
        BASE / "01_메타데이터" / "radargrams_index.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    pairs = list(zip(index["activity_id"], index["path"]))
    if args.activities:
        wanted = set(args.activities.split(","))
        pairs = [p for p in pairs if p[0] in wanted]
    lines_now = pd.read_csv(DOCS / "fourtu_apex_candidate_lines_v1.csv", dtype=ID)
    n_peaks_ref = {(a, p): n for a, p, n in zip(lines_now["activity_id"], lines_now["path"], lines_now["n_peaks"])}
    rule_now = {(a, p): r for a, p, r in zip(lines_now["activity_id"], lines_now["path"], lines_now["zero_rule"])}
    lines_old = read_csv_git(OLD_ZERO_REV, "docs/processing/fourtu_apex_candidate_lines_v1.csv")
    tz_old = {(a, p): float(t) for a, p, t in zip(lines_old["activity_id"], lines_old["path"], lines_old["t_zero_ns"])}

    cand = pd.read_csv(DOCS / "fourtu_apex_candidates_v1.csv", dtype=ID)
    cand = cand[cand["source"] == "auto_v1"]
    cand_by_line = defaultdict(list)
    for a, p, t, s in zip(cand["activity_id"], cand["path"], cand["trace"], cand["sample"]):
        cand_by_line[(a, p)].append((int(t), float(s)))
    depth_c = pd.read_csv(DOCS / "depth_4tu_v2_candidates.csv", dtype=str, keep_default_na=False)
    depth_c = depth_c[depth_c["source"] == "auto_v1"]
    hint_flag = {
        (a, p, int(t), round(float(s), 2)): ("apex_hint_unmatched" in f, st, q == "True")
        for a, p, t, s, f, st, q in zip(
            depth_c["activity_id"], depth_c["path"], depth_c["candidate_trace"],
            depth_c["candidate_sample"], depth_c["flags"], depth_c["status"], depth_c["qc_excluded"],
        )
    }

    truth = pd.read_csv(DOCS / "m210d_truth_classes.csv", dtype=ID)
    truth = truth[truth["mechanism"] == NO_PEAK].reset_index(drop=True)
    truth_acts = set(truth["activity_id"])

    line_rows, two_rows, cand_rows, truth_line_rows = [], [], [], []
    act_peak_depths = defaultdict(lambda: defaultdict(list))  # act -> variant -> nominal depths
    for n, (act, path) in enumerate(pairs, 1):
        rg = load_fourtu_radargram(BASE, act, path)
        dt, dx, f_c = rg.dt_ns, rg.dx_m, rg.f_c_hz
        inputs = prepare_depth_inputs(rg)
        t_z = inputs.zero.t_zero_ns
        row = {"activity_id": act, "path": path, "n_traces": rg.n_traces, "dt_ns": dt,
               "eps_r_used": rg.eps_r_meta, "zero_rule": inputs.zero.zero_rule, "t_zero_ns": t_z}
        if t_z is None:
            row["reason"] = "zero_failed"
            line_rows.append(row)
            continue
        S = np.asarray(inputs.branch.background_removed, dtype=np.float64)
        E = pick_signal(inputs.branch.background_removed, "envelope")
        k_lo, k_hi = fr.search_rows(t_z, f_c, dt, rg.n_samples)
        edge = int(math.ceil(gen.EDGE_M / dx))
        v = velocity_from_eps(rg.eps_r_meta)
        sigma = fr.noise_sigma(S[k_lo:k_hi])
        sigma_min = fr.row_noise_sigma_min(S[k_lo:k_hi], f_c, dt)
        per = 1e9 / f_c
        res = {}
        for rule in fr.RULES:
            for cname, cap in CAPS.items():
                pk, dg, level, lam = fr.detect(E, S, k_lo, k_hi, f_c, dt, edge, rule, cap)
                res[(rule, cname)] = (pk, dg, level, lam)
        pk0, dg0, level, _ = res[("current", "cap300")]
        k_arg = dg0["k_argmax_level"]
        floor_rows = np.nonzero(level < dg0["F"])[0]
        row.update(
            k_lo=k_lo, k_hi=k_hi,
            depth_mute_m=v * (k_lo * dt - t_z) / 2,
            L=dg0["L"],
            tau_argmax_level_ns=k_arg * dt - t_z,
            periods_argmax_after_mute=(k_arg - k_lo) * dt / per,
            argmax_in_first_period=bool((k_arg - k_lo) * dt < per),
            F_current=dg0["F"],
            F_A=res[("A_skip1", "cap300")][1]["F"],
            sigma_n=sigma,
            sigma_row_min=sigma_min,
            F_B=res[("B_noise", "cap300")][1]["F"],
            L_over_sigma=dg0["L"] / sigma if sigma > 0 else np.nan,
            Fcur_over_FB=dg0["F"] / res[("B_noise", "cap300")][1]["F"],
            frac_rows_floor_active=dg0["frac_rows_floor_active"],
            depth_first_floor_row_m=(v * ((floor_rows[0] + k_lo) * dt - t_z) / 2) if floor_rows.size else np.nan,
            n_peaks_ref=n_peaks_ref.get((act, path)),
            repro_ok=len(pk0) == n_peaks_ref.get((act, path)),
        )
        for (rule, cname), (pk, dg, _, _) in res.items():
            row[f"n_{rule}_{cname}"] = len(pk)
            row[f"raw_{rule}_{cname}"] = dg["n_raw_peaks"]
        line_rows.append(row)

        # part 3: kept auto candidates vs 5 sigma_n
        for tr, smp in cand_by_line.get((act, path), []):
            e_c = float(E[int(round(smp)), tr])
            flag = hint_flag.get((act, path, tr, round(smp, 2)))
            cand_rows.append({
                "activity_id": act, "path": path, "trace": tr, "sample": smp,
                "E_over_5sigma": e_c / (fr.K_NOISE * sigma) if sigma > 0 else np.nan,
                "E_over_5sigma_min": e_c / (fr.K_NOISE * sigma_min) if sigma_min > 0 else np.nan,
                "lambda_ratio_B_over_current": float(
                    res[("B_noise", "cap300")][3][min(int(round(smp)), k_hi - 1) - k_lo]
                    / res[("current", "cap300")][3][min(int(round(smp)), k_hi - 1) - k_lo]
                ),
                "hint_unmatched": None if flag is None else flag[0],
                "status": None if flag is None else flag[1],
                "qc_pass": None if flag is None else (flag[1] == "ok" and not flag[2]),
                "depth_nom_m": v * (smp * dt - t_z) / 2,
            })

        # part 2: two-event lines, old (v2.0) vs new zero
        if rule_now.get((act, path)) == "air_first_event" and (act, path) in tz_old:
            t_old = tz_old[(act, path)]
            k_lo_old, _ = fr.search_rows(t_old, f_c, dt, rg.n_samples)
            region = k_lo_old + math.ceil(per / dt)
            for rule in fr.RULES:
                for cname, cap in CAPS.items():
                    new_pk, new_dg, _, _ = res[(rule, cname)]
                    old_pk, old_dg, _, _ = fr.detect(E, S, k_lo_old, k_hi, f_c, dt, edge, rule, cap)
                    a_new = [p for p in new_pk if p[0] >= region]
                    a_old = [p for p in old_pk if p[0] >= region]
                    two_rows.append({
                        "activity_id": act, "path": path, "rule": rule, "cap": cname,
                        "t_zero_old_ns": t_old, "t_zero_new_ns": t_z,
                        "k_lo_old": k_lo_old, "k_lo_new": k_lo,
                        "F_old": old_dg["F"], "F_new": new_dg["F"],
                        "F_ratio_new_over_old": new_dg["F"] / old_dg["F"],
                        "n_old_region": len(a_old), "n_new_region": len(a_new),
                        "lost": len(a_old) - match_count(a_old, a_new),
                        "gained": len(a_new) - match_count(a_new, a_old),
                    })

        # part 4: GT activities with a no-peak truth (already seen)
        if act in truth_acts:
            for (rule, cname), (pk, _, _, _) in res.items():
                for k, j, _ in pk:
                    k0, _ = parabolic_peak(E[:, j], k)
                    act_peak_depths[act][f"{rule}_{cname}"].append(v * (k0 * dt - t_z) / 2)
            lam_cur = res[("current", "cap300")][3]
            for ti, tr in truth[truth["activity_id"] == act].iterrows():
                d_gt = float(tr["gt_depth_m"])
                k_w = np.arange(k_lo, k_hi)
                dn = v * (k_w * dt - t_z) / 2
                win = k_w[np.abs(dn - d_gt) <= LIMIT]
                out = {"truth_row": ti, "activity_id": act, "path": path, "gt_depth_m": d_gt,
                       "sigma_n": sigma, "F_current": dg0["F"], "L": dg0["L"],
                       "window_rows": int(win.size),
                       "window_in_search": bool(win.size > 0)}
                if win.size:
                    blk = E[win[0]: win[-1] + 1, edge: rg.n_traces - edge]
                    kk, jj = np.unravel_index(np.argmax(blk), blk.shape)
                    k_m = int(win[0] + kk)
                    lev_m = float(level[k_m - k_lo])
                    e_m = float(blk[kk, jj])
                    # Rayleigh index: (E/l) at the window maximum over the expected maximum of
                    # E/median for pure Gaussian-noise envelope over the window pixels
                    n_px = blk.size
                    ray_exp = math.sqrt(2 * math.log(n_px)) / math.sqrt(2 * math.log(2))
                    out.update(
                        rayleigh_index=(e_m / lev_m) / ray_exp,
                        Emax_win=e_m,
                        Emax_over_2F=e_m / (2 * dg0["F"]),
                        Emax_over_2level=e_m / (2 * lev_m),
                        Emax_over_5sigma=e_m / (fr.K_NOISE * sigma),
                        frac_win_rows_floor_active=float(np.mean(level[win - k_lo] < dg0["F"])),
                    )
                for (rule, cname), (pk, _, lev_r, lam_r) in res.items():
                    inwin = []
                    for k, j, D in pk:
                        k0, _ = parabolic_peak(E[:, j], k)
                        if abs(v * (k0 * dt - t_z) / 2 - d_gt) <= LIMIT:
                            inwin.append((k, j, D, k0))
                    key = f"{rule}_{cname}"
                    out[f"n_{key}"] = len(inwin)
                    out[f"maxD_{key}"] = max((p[2] for p in inwin), default=np.nan)
                    if cname == "nocap" and inwin:
                        n_fa = n_fa5 = n_acc = n_acc_win = 0
                        for k, j, D, k0 in inwin:
                            floor_active = bool(lev_r[k - k_lo] < res[("current", "cap300")][1]["F"])
                            above5 = bool(E[k, j] >= fr.K_NOISE * sigma)
                            n_fa += floor_active
                            n_fa5 += floor_active and above5
                            ja, ka = gen.crest_apex(E, j, k0, dt, dx)
                            if not edge <= ja < rg.n_traces - edge or ka < k_lo:
                                continue
                            vq, nl, nr, why = gen.accept(E, ja, ka, dt, dx, t_z)
                            if vq is None:
                                continue
                            n_acc += 1
                            n_acc_win += abs(v * (ka * dt - t_z) / 2 - d_gt) <= LIMIT
                        out[f"floor_active_{key}"] = n_fa
                        out[f"floor_active_5sigma_{key}"] = n_fa5
                        out[f"accepted_{key}"] = n_acc
                        out[f"accepted_in_window_{key}"] = n_acc_win
                truth_line_rows.append(out)
        if n % 50 == 0 or n == len(pairs):
            print(f"[{n}/{len(pairs)}] {act}/{path} elapsed {time.perf_counter() - started:.0f} s", flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    suffix = "_subset" if args.activities else ""
    lines_df = pd.DataFrame(line_rows)
    lines_df.to_csv(DATA / f"m210b_lines{suffix}.csv", index=False, float_format="%.6g")
    two_df = pd.DataFrame(two_rows)
    two_df.to_csv(DATA / f"m210b_two_event{suffix}.csv", index=False, float_format="%.6g")
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(DATA / f"m210b_candidates_vs_noise{suffix}.csv", index=False, float_format="%.6g")
    tl = pd.DataFrame(truth_line_rows)
    tl.to_csv(DATA / f"m210b_truth_lines{suffix}.csv", index=False, float_format="%.6g")

    # ------------------------------------------------ part 4 classes (fixed in doc 3-1)
    t_rows = []
    for ti, tr in truth.iterrows():
        g = tl[tl["truth_row"] == ti] if len(tl) else tl
        if g.empty:
            continue

        def anyp(col: str) -> bool:
            return bool(col in g and (g[col].fillna(0) > 0).any())

        if anyp("n_current_cap300"):
            # a current peak lies in the window by its own (refined) depth; m210d used the
            # depth after the crest move, so the crest move took it out of the window
            cls = "crest_moved_out"
        elif anyp("n_current_nocap"):
            cls = "cap300"
        elif anyp("floor_active_5sigma_C_none_nocap"):
            cls = "floor"
        elif anyp("floor_active_C_none_nocap"):
            cls = "floor_noise_level"
        else:
            cls = "no_energy"
        act = tr["activity_id"]
        t_rows.append({
            "activity_id": act, "site": tr["site"], "gt_depth_m": tr["gt_depth_m"],
            "status": tr["status"], "n_lines": len(g), "class": cls,
            "lines_window_in_search": int(g["window_in_search"].sum()),
            "max_Emax_over_2F": g.get("Emax_over_2F", pd.Series(dtype=float)).max(),
            "max_Emax_over_2level": g.get("Emax_over_2level", pd.Series(dtype=float)).max(),
            "max_Emax_over_5sigma": g.get("Emax_over_5sigma", pd.Series(dtype=float)).max(),
            "max_rayleigh_index": g.get("rayleigh_index", pd.Series(dtype=float)).max(),
            "median_frac_win_floor_active": g.get("frac_win_rows_floor_active", pd.Series(dtype=float)).median(),
            **{f"lines_with_peak_{r}_{c}": int((g[f"n_{r}_{c}"] > 0).sum()) for r in fr.RULES for c in CAPS},
            **{f"accepted_in_window_{r}_nocap": int(g.get(f"accepted_in_window_{r}_nocap", pd.Series(dtype=float)).fillna(0).sum()) for r in fr.RULES},
            **{f"chance_coverage_{r}_nocap": round(coverage(act_peak_depths[act][f"{r}_nocap"]), 3) for r in fr.RULES},
            **{f"chance_coverage_{r}_cap300": round(coverage(act_peak_depths[act][f"{r}_cap300"]), 3) for r in fr.RULES},
        })
    truth_df = pd.DataFrame(t_rows)
    if not args.activities:
        truth_df.to_csv(DOCS / "m210b_truth_no_peak.csv", index=False, float_format="%.4f")

    # ------------------------------------------------ summary
    ld = lines_df[lines_df["t_zero_ns"].notna()]
    summ = {
        "note": "already-seen 4TU data, exploration only (M2-10 b); not a verdict, not for tuning",
        "n_lines": int(len(lines_df)),
        "reproduction_current_cap300_equals_n_peaks": int(ld["repro_ok"].sum()),
        "part1_labelfree": {
            "argmax_level_in_first_period_after_mute": int(ld["argmax_in_first_period"].sum()),
            "periods_argmax_after_mute_q10_50_90": [round(float(x), 2) for x in ld["periods_argmax_after_mute"].quantile([0.1, 0.5, 0.9])],
            "frac_rows_floor_active_q10_50_90": [round(float(x), 3) for x in ld["frac_rows_floor_active"].quantile([0.1, 0.5, 0.9])],
            "depth_first_floor_row_m_q10_50_90": [round(float(x), 3) for x in ld["depth_first_floor_row_m"].quantile([0.1, 0.5, 0.9])],
            "L_over_sigma_q10_50_90": [round(float(x), 1) for x in ld["L_over_sigma"].quantile([0.1, 0.5, 0.9])],
            "Fcur_over_FB_q10_50_90": [round(float(x), 3) for x in ld["Fcur_over_FB"].quantile([0.1, 0.5, 0.9])],
            "lines_Fcur_gt_FB": int((ld["Fcur_over_FB"] > 1).sum()),
            "n_peaks_median": {f"{r}_{c}": float(ld[f"n_{r}_{c}"].median()) for r in fr.RULES for c in CAPS},
            "lines_cap300_hit": {r: int((ld[f"raw_{r}_cap300"] >= fr.NUM_PEAKS).sum()) for r in fr.RULES},
        },
        "part2_two_event": {},
        "part3_candidates": {},
        "part4_truth_no_peak": {},
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    if len(two_df):
        for (rule, cname), g in two_df.groupby(["rule", "cap"]):
            summ["part2_two_event"][f"{rule}_{cname}"] = {
                "lines": int(len(g)),
                "n_old_region": int(g["n_old_region"].sum()),
                "lost": int(g["lost"].sum()),
                "gained": int(g["gained"].sum()),
                "lines_with_change": int(((g["lost"] + g["gained"]) > 0).sum()),
                "F_ratio_q10_50_90": [round(float(x), 3) for x in g["F_ratio_new_over_old"].quantile([0.1, 0.5, 0.9])],
                "F_ratio_min_max": [round(float(g["F_ratio_new_over_old"].min()), 3), round(float(g["F_ratio_new_over_old"].max()), 3)],
            }
        p10 = two_df[(two_df["activity_id"] == "06.6") & (two_df["path"] == "Path10") & (two_df["cap"] == "cap300")]
        summ["part2_two_event"]["06.6_Path10"] = p10[["rule", "F_old", "F_new", "n_old_region", "n_new_region", "lost", "gained"]].round(3).to_dict(orient="records")
    if len(cand_df):
        c = cand_df
        below = c["E_over_5sigma"] < 1
        summ["part3_candidates"] = {
            "n": int(len(c)),
            "below_5sigma": int(below.sum()),
            "below_5sigma_hint_unmatched": int((below & (c["hint_unmatched"] == True)).sum()),  # noqa: E712
            "above_5sigma_hint_unmatched": int((~below & (c["hint_unmatched"] == True)).sum()),  # noqa: E712
            "hint_unmatched_total": int((c["hint_unmatched"] == True).sum()),  # noqa: E712
            "E_over_5sigma_q01_05_50": [round(float(x), 3) for x in c["E_over_5sigma"].quantile([0.01, 0.05, 0.5])],
            "lambda_B_gt_current_at_candidate": int((c["lambda_ratio_B_over_current"] > 1 + 1e-12).sum()),
            "below_5sigma_qc_pass": int((below & (c["qc_pass"] == True)).sum()),  # noqa: E712
            "above_5sigma_qc_pass": int((~below & (c["qc_pass"] == True)).sum()),  # noqa: E712
            "below_5sigma_min": int((c["E_over_5sigma_min"] < 1).sum()),
            "below_5sigma_min_qc_pass": int(((c["E_over_5sigma_min"] < 1) & (c["qc_pass"] == True)).sum()),  # noqa: E712
            "below_5sigma_depth_nom_q10_50_90": [round(float(x), 3) for x in c[below]["depth_nom_m"].quantile([0.1, 0.5, 0.9])],
            "above_5sigma_depth_nom_q10_50_90": [round(float(x), 3) for x in c[~below]["depth_nom_m"].quantile([0.1, 0.5, 0.9])],
        }
    if len(truth_df):
        summ["part4_truth_no_peak"] = {
            "n": int(len(truth_df)),
            "class_counts": dict(Counter(truth_df["class"])),
            "floor_and_accepted_in_window_C": int(((truth_df["class"] == "floor") & (truth_df["accepted_in_window_C_none_nocap"] > 0)).sum()),
            "recovered_peak_by_rule_cap300": {r: int((truth_df[f"lines_with_peak_{r}_cap300"] > 0).sum()) for r in fr.RULES},
            "recovered_peak_by_rule_nocap": {r: int((truth_df[f"lines_with_peak_{r}_nocap"] > 0).sum()) for r in fr.RULES},
            "recovered_accepted_in_window_nocap": {r: int((truth_df[f"accepted_in_window_{r}_nocap"] > 0).sum()) for r in fr.RULES},
        }
    name = "m210b_floor_diagnosis_subset.json" if args.activities else "m210b_floor_diagnosis.json"
    (DOCS / name).write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
