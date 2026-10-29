"""Theoretical t0/v table and hyperbola-fit bias study for the synthetic model set.

Tables (Markdown, written to data/synthetic_gprmax/theory_tables.md and printed):
  A  ground truth per model: v, t0_top, t0_center, air two-way time, apex ray time
  B  radius bias: point-diffractor hyperbola t=(2/v')sqrt(d'^2+(x-x0)^2) fitted to the exact zero-offset
     cylinder curve t=(2/v)(sqrt((d+R)^2+(x-x0)^2)-R), h=0, apertures 0.5 m and 1.0 m
  C  realistic geometry bias: ray times with antenna offset and air-gap refraction, time-zero at the
     emission (h=0) or at the geometric surface reflection (h>0); fitted with the point model and with
     the radius model (R known); aperture 1.0 m
  D  measured (from <id>.json written by export_dataset.py): apex pick, wavelet lag, depth from t0 under
     three time-zero conventions, artefact checks

Run: conda run -n gprmax-acceleration python -I scripts/synthetic/theory_table.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ray_geometry import C0_M_PER_NS, surface_reflection_time, trace_reflection_times  # noqa: E402

OUT = HERE.parent.parent / "data" / "synthetic_gprmax"


def fit_point(x, t, x0):
    """Least squares in t for t = (2/v) sqrt(d^2 + (x-x0)^2), x0 known. Returns v, d."""

    def res(p):
        v, d = p
        return 2.0 / v * np.sqrt(d * d + (x - x0) ** 2) - t

    v_init = 0.1
    d_init = v_init * t.min() / 2
    sol = least_squares(res, [v_init, d_init], bounds=([0.01, 0.0], [0.3, 10.0]), xtol=1e-14, ftol=1e-14)
    return sol.x


def fit_radius(x, t, x0, radius):
    """Least squares for t = (2/v)(sqrt((d+R)^2+(x-x0)^2) - R) with R known. Returns v, d_top."""

    def res(p):
        v, d = p
        return 2.0 / v * (np.sqrt((d + radius) ** 2 + (x - x0) ** 2) - radius) - t

    sol = least_squares(res, [0.1, 0.5], bounds=([0.01, 0.0], [0.3, 10.0]), xtol=1e-14, ftol=1e-14)
    return sol.x


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    specs = json.loads((HERE / "model_specs.json").read_text(encoding="utf-8"))
    parts = []

    # A
    rows = []
    for s in specs:
        v = s["velocity_m_per_ns"]
        h = round(s["y_antenna_m"] - s["y_surface_m"], 6)
        apex = trace_reflection_times([s["pipe_center_x_m"]], s["antenna_offset_m"], h, s["pipe_center_x_m"],
                                      s["pipe_top_depth_m"], s["pipe_radius_m"], v)[0]
        rows.append([s["model_id"], s["pipe_kind"], s["pipe_outer_diameter_m"], s["pipe_top_depth_m"],
                     f"{s['pipe_top_depth_m'] + s['pipe_radius_m']:.3f}", s["eps_r"], f"{v:.4f}",
                     f"{s['center_freq_hz'] / 1e6:.0f}", h, f"{2 * s['pipe_top_depth_m'] / v:.3f}",
                     f"{2 * (s['pipe_top_depth_m'] + s['pipe_radius_m']) / v:.3f}", f"{2 * h / C0_M_PER_NS:.3f}",
                     f"{apex:.3f}"])
    parts.append("### A. 정답과 이론 t0\n\n" + md_table(
        ["id", "관", "D (m)", "d_top (m)", "d_center (m)", "εr", "v (m/ns)", "f (MHz)", "h (m)",
         "t0_top (ns)", "t0_center (ns)", "공기 왕복 (ns)", "꼭짓점 광선시간 (ns)"], rows))

    # B: radius bias, unique (D, d_top) pairs, soil only, zero offset; v enters only as a scale
    rows = []
    seen = set()
    for s in specs:
        key = (s["pipe_outer_diameter_m"], s["pipe_top_depth_m"])
        if key in seen:
            continue
        seen.add(key)
        r, d = s["pipe_radius_m"], s["pipe_top_depth_m"]
        v = 0.1
        out = [f"{2 * r:.2f}", d]
        for aperture in (0.5, 1.0):
            x = np.arange(-aperture, aperture + 1e-9, 0.02)
            t = 2 / v * (np.sqrt((d + r) ** 2 + x ** 2) - r)
            vf, df = fit_point(x, t, 0.0)
            out += [f"{vf / v:.4f}", f"{df:.3f}", f"{df - d:+.3f}"]
        rows.append(out)
    rows.sort(key=lambda r: (float(r[0]), r[1]))
    parts.append("### B. 점산란 쌍곡선을 원통 곡선에 맞췄을 때의 편향 (공기층 0, 오프셋 0)\n\n" + md_table(
        ["D (m)", "d_top (m)", "v'/v (±0.5 m)", "d' (m)", "d'−d_top", "v'/v (±1.0 m)", "d' (m)", "d'−d_top"],
        rows))

    # C: realistic geometry
    rows = []
    for s in specs:
        v = s["velocity_m_per_ns"]
        h = round(s["y_antenna_m"] - s["y_surface_m"], 6)
        x0, r, d = s["pipe_center_x_m"], s["pipe_radius_m"], s["pipe_top_depth_m"]
        x = x0 + np.arange(-1.0, 1.0 + 1e-9, 0.02)
        t_ray = trace_reflection_times(x, s["antenna_offset_m"], h, x0, d, r, v)
        tz = surface_reflection_time(s["antenna_offset_m"], h) if h > 0 else 0.0
        t = t_ray - tz
        vp, dp = fit_point(x, t, x0)
        vr, dr = fit_radius(x, t, x0, r)
        t0_apex = t.min()
        rows.append([s["model_id"], h, f"{2 * r:.2f}", d, f"{t0_apex:.3f}", f"{v * t0_apex / 2:.3f}",
                     f"{vp / v:.4f}", f"{dp:+.3f}", f"{dp - d:+.3f}", f"{vr / v:.4f}", f"{dr - d:+.3f}"])
    parts.append("### C. 실제 배치(오프셋 0.10 m, 공기층 굴절)에서의 피팅 편향 (구간 ±1.0 m)\n\n"
                 "time-zero: h=0은 송신 시각, h>0은 지표 반사 기하 시각. 피팅 모델은 오프셋을 무시한 "
                 "영오프셋 식.\n\n" + md_table(
                     ["id", "h (m)", "D (m)", "d_top (m)", "꼭짓점 t (ns)", "v·t/2 (m)", "점 모델 v'/v", "점 모델 d'",
                      "점 모델 d'−d_top", "반경 모델 v'/v", "반경 모델 d−d_top"], rows))

    # D: measured
    rows = []
    for s in specs:
        p = OUT / f"{s['model_id']}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        a, de, ck, vc = m["apex"], m["depth_from_t0_effective_m"], m["checks"], m["velocity_check"]
        fmt = lambda val: "—" if val is None else f"{val:.3f}"  # noqa: E731
        rows.append([m["model_id"], f"{a['expected_envelope_peak_ns']:.3f}", f"{a['envelope_peak_ns']:.3f}",
                     f"{a['wavelet_lag_ns']:+.3f}", "예" if a["envelope_peak_at_window_edge"] else "아니오",
                     f"{a['signed_peak_polarity']:+d}", fmt(de["source_peak"]), fmt(de["direct_wave_peak"]),
                     fmt(de["surface_reflection_peak"]), m["pipe_top_depth_m"],
                     f"{max(ck['bg_lateral_diff_rel_x0.4_x2.6']):.1e}", f"{ck['pre_arrival_residual_rel']:.1e}",
                     "—" if ck["bottom_pml_window_envelope_rel"] is None else f"{ck['bottom_pml_window_envelope_rel']:.1e}",
                     f"{vc['rms_after_linear_ns']:.3f}"])
    if rows:
        parts.append("### D. 시뮬레이션 실측 (꼭짓점 트레이스 x=1.5 m, 배경 제거 후 포락선)\n\n" + md_table(
            ["id", "예상 피크 (ns)", "실측 피크 (ns)", "wavelet 지연 (ns)", "창 경계", "극성",
             "d: 송신 기준 (m)", "d: 직접파 기준 (m)", "d: 지표반사 기준 (m)", "정답 d_top (m)",
             "측방 PML 지표", "도달 전 잔여", "바닥 PML 창", "픽 잔차 RMS (ns)"], rows))

    # F: end-to-end fit on measured envelope picks (what a picking + fitting pipeline would see)
    rows = []
    for s in specs:
        p = OUT / f"{s['model_id']}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        picks = m["trace_envelope_pick_ns"]
        x_all = m["trace_x_m_first"] + np.arange(m["n_traces"]) * m["dx_m"]
        keep = [i for i, val in enumerate(picks) if val is not None]
        x = x_all[keep]
        tp = np.array([picks[i] for i in keep])
        v, r, d, x0 = m["velocity_m_per_ns"], m["pipe_radius_m"], m["pipe_top_depth_m"], m["pipe_center_x_m"]
        tzc = m["time_zero_candidates_ns"]
        conv = "surface_reflection_peak" if m["air_gap_m"] > 0 else "direct_wave_peak"
        t = tp - tzc[conv]
        vp, dp = fit_point(x, t, x0)
        vr, dr = fit_radius(x, t, x0, r)
        # same, but with the exact emission time-zero plus the measured apex wavelet lag (ideal calibration)
        t_ideal = tp - (tzc["source_peak"] + m["apex"]["wavelet_lag_ns"])
        if m["air_gap_m"] > 0:
            t_ideal = t_ideal - m["surface_reflection_geom_ns"]
        vi, di = fit_point(x, t_ideal, x0)
        vri, dri = fit_radius(x, t_ideal, x0, r)
        rows.append([m["model_id"], len(keep), conv.replace("_peak", ""), f"{vp / v:.3f}", f"{dp - d:+.3f}",
                     f"{vr / v:.3f}", f"{dr - d:+.3f}", f"{vi / v:.3f}", f"{di - d:+.3f}", f"{vri / v:.3f}",
                     f"{dri - d:+.3f}"])
    if rows:
        parts.append("### F. 실측 포락선 픽에 피팅 (|x−x0| ≤ 1.0 m, x0 고정)\n\n"
                     "왼쪽: 데이터에서 잰 time-zero(h=0 직접파 피크, h>0 지표 반사 피크). 오른쪽: 이상적 보정"
                     "(송신 피크 + 꼭짓점 wavelet 지연, h>0은 지표 반사 기하 시간까지 뺌).\n\n" + md_table(
                         ["id", "픽 수", "time-zero", "점 v'/v", "점 d'−d_top", "반경 v'/v", "반경 d−d_top",
                          "이상 점 v'/v", "이상 점 d'−d_top", "이상 반경 v'/v", "이상 반경 d−d_top"], rows))

    # Sensitivity
    rows = []
    for eps in (4.0, 9.0, 16.0):
        v = C0_M_PER_NS / math.sqrt(eps)
        for d in (0.3, 0.6, 1.0, 1.5):
            t0 = 2 * d / v
            dd_eps = d * (1 / math.sqrt(1.1) - 1)  # eps overestimated by 10 %
            dd_tz = v * 0.5 / 2  # time-zero error 0.5 ns
            dd_smp = v * 0.098 / 2  # one 4TU sample
            rows.append([eps, d, f"{t0:.2f}", f"{dd_eps:+.3f}", f"{dd_tz:.3f}", f"{dd_smp:.4f}"])
    parts.append("### E. 민감도 (d = v·t0/2)\n\n" + md_table(
        ["εr", "d (m)", "t0 (ns)", "εr +10% → Δd (m)", "time-zero 0.5 ns → Δd (m)", "1 샘플 0.098 ns → Δd (m)"],
        rows))

    text = "\n\n".join(parts) + "\n"
    (OUT / "theory_tables.md").write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console default (cp949) cannot encode the Greek letters
    print(text)


if __name__ == "__main__":
    main()
