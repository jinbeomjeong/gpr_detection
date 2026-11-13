"""Scale survey for the detection-image (L1) design (M3-1/M3-2 spec draft).

Collects, without heavy computation:
  1. AIHub box sizes per class (gain copies removed) and the AIHub vertical
     carrier period in px (from aihub_wavelet_length.json).
  2. Grid sizes (traces, samples, dt, dx) of 4TU, Guangzhou, GPR-SD lines.
  3. 4TU true-apex tracked apertures from the blinded review (type T).
  4. Dominant vertical period in samples for a few 4TU / GPR-SD lines
     (mean trace amplitude spectrum, memory-mapped subsets only).
  5. Analytic hyperbola box sizes per dataset configuration, native and at
     AIHub-matched scale, plus whole-line resize (alternative A) sizes.
  6. 4TU ground-truth horizontal spacing of neighbouring utilities (merge rule).

Output: docs/processing/detection_image_scale_survey_v1.json and a text dump.
Read-only on all inputs. Single process, no random numbers.
"""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\workspace\gpr_detection")
GPR = Path(r"C:\workspace\GPR_Data")
FOURTU = GPR / "4TU_네덜란드_매설관GPR_가공본_261015"
GZ = GPR / "광저우대_중국_관로GPR_가공본_261016"
SD = GPR / "GPR-SD_중국USTC_도로GPR_가공본_261016"
OUT = ROOT / "docs" / "processing" / "detection_image_scale_survey_v1.json"
# Note: "fourtu_spectra" in the output is the raw DC-removed estimate (biased long by the
# zero band and wow); the L1 analysis uses detection_image_period_check_v1.json instead.


def quant(values, ps=(5, 10, 25, 50, 75, 90, 95)):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    out = {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps}
    out.update(n=int(a.size), min=round(float(a.min()), 3), max=round(float(a.max()), 3))
    return out


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 1. AIHub
def aihub_boxes():
    rows = read_csv(ROOT / "data" / "audit" / "aihub_71327" / "objects.csv")
    seen = set()
    per = defaultdict(list)
    for r in rows:
        key_img = re.sub(r"_g[0-9.]+_", "_", r["filename"])
        x1, y1, x2, y2 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        key = (r["name"], key_img, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
        if key in seen:
            continue
        seen.add(key)
        per[r["name"]].append((x1, y1, x2, y2))
    out = {}
    for name, boxes in per.items():
        b = np.asarray(boxes)
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        edge = (b[:, 0] <= 1.0) | (b[:, 2] >= 255.0)
        out[name] = {
            "n_unique": int(len(b)),
            "w_px": quant(w),
            "h_px": quant(h),
            "w_over_h": quant(w / np.maximum(h, 1e-6)),
            "ymin_px": quant(b[:, 1]),
            "ymax_px": quant(b[:, 3]),
            "touch_lr_edge_frac": round(float(edge.mean()), 4),
        }
    return out


def aihub_period():
    j = json.loads((ROOT / "analysis" / "processing" / "aihub_wavelet_length.json").read_text(encoding="utf-8"))
    return j["aihub"]["hyperbola_apex"]["P_px"]


# ---------------------------------------------------------------- spectra
def dominant_period_samples(arr, fmin_cyc=0.005):
    """Mean amplitude spectrum over traces after DC removal; returns period in samples."""
    a = np.asarray(arr, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n = a.shape[0]
    spec = np.abs(np.fft.rfft(a, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(n, d=1.0)
    ok = f >= fmin_cyc
    fp = f[ok][np.argmax(spec[ok])]
    return 1.0 / fp, float(fp)


# ---------------------------------------------------------------- 2. grids
def fourtu_grid():
    rows = read_csv(ROOT / "docs" / "processing" / "fourtu_lines_v1.csv")
    nt = [int(r["n_traces"]) for r in rows]
    dt = Counter(r["dt_ns"] for r in rows)
    ns = Counter(r["n_samples"] for r in rows)
    eps = [fnum(r["eps_r_used"]) for r in rows]
    big = {thr: sum(1 for n in nt if n > thr) for thr in (256, 512, 1000, 1500, 2560, 5000)}
    longest = sorted(rows, key=lambda r: -int(r["n_traces"]))[:5]
    return rows, {
        "n_lines": len(rows),
        "n_traces": quant(nt),
        "n_traces_gt": big,
        "dt_ns_counts": dict(dt),
        "n_samples_counts": dict(ns),
        "eps_r_used": quant(eps),
        "longest": [(r["activity_id"], r["path"], int(r["n_traces"]), r["dt_ns"]) for r in longest],
    }


def gz_grid():
    rows = [r for r in read_csv(GZ / "01_메타데이터" / "radargrams_index.csv") if r["target"] == "pipe"]
    groups = defaultdict(list)
    for r in rows:
        key = (r["format"], r["antenna_MHz"], r["n_samples"], r["time_window_ns"], r["trace_dx_m"])
        groups[key].append(r)
    g_out = []
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        dts = [fnum(r["dt_ns"]) for r in rs]
        fpk = [fnum(r["spectrum_peak_MHz"]) for r in rs]
        p_samp = [1.0 / (fp * 1e-3 * d) for fp, d in zip(fpk, dts) if fp and d]
        g_out.append({
            "format": key[0], "antenna_MHz": key[1], "n_samples": key[2], "time_window_ns": key[3],
            "trace_dx_m": key[4], "n_files": len(rs), "dt_ns": quant(dts, (50,)),
            "n_traces": quant([int(r["n_traces"]) for r in rs], (10, 50, 90)),
            "spectrum_peak_MHz": quant(fpk, (10, 50, 90)),
            "period_samples_from_peak": quant(p_samp, (10, 50, 90)),
        })
    return rows, {
        "n_pipe_files": len(rows),
        "n_traces_all": quant([int(r["n_traces"]) for r in rows]),
        "n_samples_counts": dict(Counter(r["n_samples"] for r in rows)),
        "format_counts": dict(Counter(r["format"] for r in rows)),
        "groups_top": g_out[:14],
    }


def sd_grid():
    rows = read_csv(SD / "01_메타데이터" / "radargrams_index.csv")
    out = {"lines": [(r["file_id"], int(r["n_samples"]), int(r["n_traces"])) for r in rows]}
    per = {}
    for fid in ("CementRoad1_1", "AsphaltRoad1_1", "CementRoad2_1"):
        a = np.load(SD / "05_radargrams_npy" / f"{fid}.npy", mmap_mode="r")
        sub = np.asarray(a[:, 1000:4000:3])  # 1000 traces, small read
        p, fcyc = dominant_period_samples(sub)
        per[fid] = {
            "shape": list(a.shape), "period_samples": round(p, 2), "f_cyc_per_sample": round(fcyc, 4),
            "f_MHz_hypA": round(fcyc / 0.3022 * 1e3, 1), "f_MHz_hypB": round(fcyc / 0.2553 * 1e3, 1),
        }
    out["spectra"] = per
    return rows, out


def fourtu_spectra(rows):
    picks = {}
    for dt in ("0.097", "0.098", "0.195"):
        r = next(r for r in rows if r["dt_ns"] == dt)
        picks[dt] = r
    out = {}
    for dt, r in picks.items():
        a = np.load(FOURTU / "05_radargrams_npy" / r["activity_id"] / f"{r['path']}.npy", mmap_mode="r")
        sub = np.asarray(a[:, : min(a.shape[1], 600)])
        p, fcyc = dominant_period_samples(sub)
        out[dt] = {
            "line": f"{r['activity_id']}/{r['path']}", "shape": list(a.shape),
            "period_samples": round(p, 2), "f_MHz": round(fcyc / float(dt) * 1e3, 1),
        }
    return out


# ---------------------------------------------------------------- 3. 4TU true apexes
def fourtu_true_apexes():
    rows = [r for r in read_csv(ROOT / "docs" / "processing" / "m29c_apex_review.csv") if r["type"] == "T"]
    return {
        "n": len(rows),
        "d_cand_m": quant([fnum(r["d_cand_m"]) for r in rows]),
        "aperture_m": quant([fnum(r["aperture_m"]) for r in rows]),
        "n_picks": quant([fnum(r["n_picks"]) for r in rows]),
        "apex_sample": quant([fnum(r["sample"]) for r in rows]),
        "tau_ns": quant([fnum(r["tau_ns"]) for r in rows]),
    }


# ---------------------------------------------------------------- 6. GT spacing
def fourtu_gt_spacing():
    rows = read_csv(FOURTU / "01_메타데이터" / "utilities_groundtruth.csv")
    groups = defaultdict(list)
    for r in rows:
        x = fnum(r["distance_m"])
        d = fnum(r["depth_m"])
        if x is None:
            continue
        groups[(r["activity_id"], r["trench_no"])].append((x, d))
    dx_nn, pairs = [], []
    for pts in groups.values():
        pts.sort()
        for (x1, d1), (x2, d2) in zip(pts, pts[1:]):
            dx_nn.append(x2 - x1)
            pairs.append((x2 - x1, None if d1 is None or d2 is None else abs(d2 - d1)))
    close = [p for p in pairs if p[0] <= 0.30]
    close_same_depth = [p for p in close if p[1] is not None and p[1] <= 0.10]
    return {
        "n_neighbour_pairs": len(pairs),
        "dx_neighbour_m": quant(dx_nn),
        "frac_dx_le_0p10": round(sum(1 for v in dx_nn if v <= 0.10) / len(dx_nn), 4),
        "frac_dx_le_0p30": round(len(close) / len(dx_nn), 4),
        "n_dx_le_0p30_and_dd_le_0p10": len(close_same_depth),
        "frac_dx_le_0p30_and_dd_le_0p10": round(len(close_same_depth) / len(dx_nn), 4),
    }


# ---------------------------------------------------------------- 5. analytic boxes
def hyperbola_box(d, a, v, dt, dx, period_ns):
    """Box of a point-scatterer hyperbola: half-width a [m], 1.5 periods above apex, 1.5 below wing end."""
    drop_ns = (2.0 / v) * (math.sqrt(d * d + a * a) - d)
    h_ns = drop_ns + 3.0 * period_ns
    return {
        "drop_ns": round(drop_ns, 3), "drop_periods": round(drop_ns / period_ns, 2),
        "h_periods": round(h_ns / period_ns, 2),
        "w_traces": int(round(2 * a / dx)) + 1, "h_samples": int(round(h_ns / dt)),
    }


def configs(p_aihub_px, sd_period, fourtu_period):
    """Configurations with periods measured after steps (1)-(4).

    Periods come from detection_image_period_check_v1.json (p50): 4TU 2.51 ns
    (whole line and apex window), Guangzhou 400 MHz 30 ns 2.93 ns / 50 ns
    2.11 ns, GPR-SD 51.2 samples (pipe-mark windows, dt-free).
    """
    pc = json.loads((ROOT / "docs" / "processing" / "detection_image_period_check_v1.json").read_text(encoding="utf-8"))
    p4 = pc["fourtu"]["period_ns"]["bg_whole"]["p50"]
    g30 = pc["guangzhou"]["30.0"]["bg_whole_padded_ns"]["p50"]
    g50 = pc["guangzhou"]["50.0"]["bg_whole_padded_ns"]["p50"]
    psd = pc["gprsd"]["mark_window_period_samples"]["p50"]
    # (name, n_samples, dt_ns, dx_m, v_m_per_ns, period_samples, note)
    cfg = []
    for dt in ("0.098", "0.195"):
        cfg.append((f"4TU dt {dt}", 512, float(dt), 0.02, 0.1, p4 / float(dt), "v 0.1 = eps 9 meta (unverified)"))
    cfg.append(("GZ IDS 400MHz 30ns", 512, 30 / 512, 0.024, 0.1, g30 / (30 / 512), "v 0.10 assumed (README)"))
    cfg.append(("GZ IDS 400MHz 50ns", 512, 50 / 512, 0.024, 0.1, g50 / (50 / 512), "v 0.10 assumed (README)"))
    # GPR-SD: only v*dt/2 per sample (0.012 m, rough curvature fit) is known; write it as dt 0.3022 with matching v.
    cfg.append(("GPR-SD hypA", 321, 0.3022, 0.03, 2 * 0.012 / 0.3022, psd, "v*dt/2 = 0.012 m/sample (rough fit), dt-free in samples"))
    return cfg


def analytic(cfg, gz_info, p_aihub_px, w_aihub_tpipe):
    # Periods come from cfg (measured after steps (1)-(4)); gz_info is kept for the signature only.
    out = []
    for name, ns, dt, dx, v, per_s, note in cfg:
        period_ns = per_s * dt
        lam_m = v * period_ns  # wavelength in the ground [m]
        s_y = per_s / p_aihub_px  # samples per L1 px so that the carrier period = AIHub period
        s_x_iso = s_y * dt * v / 2.0 / dx  # traces per px so that m/px is equal on both axes (depth units)
        m_per_px = s_x_iso * dx
        rows = []
        for d in (0.3, 0.7, 1.5):
            for a in (0.35, 0.6, 0.8):
                b = hyperbola_box(d, a, v, dt, dx, period_ns)
                rows.append({
                    "d_m": d, "a_m": a, **b,
                    "w_periods_depthunits": round(2 * a / (lam_m / 2), 2),
                    "native_w_over_h": round(b["w_traces"] / b["h_samples"], 3),
                    "iso_w_px": round(b["w_traces"] / s_x_iso, 1),
                    "iso_h_px": round(b["h_samples"] / s_y, 1),
                    "iso_w_over_h": round((b["w_traces"] / s_x_iso) / (b["h_samples"] / s_y), 3),
                    "fresnel_r_m": round(math.sqrt(lam_m * d / 2.0), 3),
                })
        zoom = {}
        for p_star in (4.06, 6.0, 8.0):
            z = p_star / p_aihub_px
            zoom[str(p_star)] = {
                "s_y": round(s_y / z, 3), "s_x": round(s_x_iso / z, 3),
                "image_h_px": round(ns / (s_y / z), 1),
                "tile256_traces": round(256 * s_x_iso / z, 1), "tile256_m": round(256 * s_x_iso / z * dx, 2),
            }
        out.append({
            "config": name, "note": note, "n_samples": ns, "dt_ns": round(dt, 5), "dx_m": dx,
            "v_m_per_ns": round(v, 4), "period_samples": round(per_s, 2), "period_ns": round(period_ns, 3),
            "lambda_m": round(lam_m, 3),
            "s_y_samples_per_px": round(s_y, 3), "image_h_px_full_window": round(ns / s_y, 1),
            "s_x_iso_traces_per_px": round(s_x_iso, 3), "m_per_px": round(m_per_px, 4),
            "native_depth_m_per_sample": round(dt * v / 2, 5),
            "native_aspect_vertical_stretch_vs_iso": round(dx / (dt * v / 2), 2),
            "overlap_px_for_2m": math.ceil(2.0 / m_per_px),
            "zoom": zoom,
            "boxes": rows,
        })
    return out


def resize_whole(nt_list, ns, w_traces):
    """Alternative A: whole line letterboxed into 256 (longest side). Hyperbola width in px."""
    res = []
    for nt in nt_list:
        f = max(nt, ns) / 256.0
        res.append(w_traces / f)
    return quant(res, (5, 10, 25, 50, 75, 90, 95))


def main():
    out = {}
    ai = aihub_boxes()
    p_px = aihub_period()
    out["aihub_boxes"] = ai
    out["aihub_P_px"] = p_px
    w_tp = ai["tpipe"]["w_px"]["p50"]

    f_rows, f_info = fourtu_grid()
    out["fourtu_grid"] = f_info
    out["fourtu_spectra"] = fourtu_spectra(f_rows)
    out["fourtu_true_apex"] = fourtu_true_apexes()
    out["fourtu_gt_spacing"] = fourtu_gt_spacing()

    g_rows, g_info = gz_grid()
    out["gz_grid"] = g_info
    s_rows, s_info = sd_grid()
    out["gprsd_grid"] = s_info

    cfg = configs(p_px["p50"], s_info["spectra"], out["fourtu_spectra"])
    out["analytic"] = analytic(cfg, g_info, p_px["p50"], w_tp)

    # Alternative A: hyperbola of 81 traces (4TU a=0.8 m), 67 traces (GZ), 54 traces (SD)
    out["alt_A_whole_line_w_px"] = {
        "4TU_81tr": resize_whole([int(r["n_traces"]) for r in f_rows], 512, 81),
        "GZ_IDS_67tr": resize_whole([int(r["n_traces"]) for r in g_rows if r["format"] == "IDS .dt"], 512, 67),
        "GPRSD_54tr": resize_whole([int(r["n_traces"]) for r in s_rows], 321, 54),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
