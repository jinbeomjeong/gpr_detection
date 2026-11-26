"""M3-6 D2m (added after the D2 runs, report only): migration-focusing scan for g = dx / dz.

Why: the automatic picker's QC-passing population (D2) contains quasi-flat, undulating events, so
its dz distribution depends on the QC choice. A Stolt (f-k) constant-velocity migration scan does
not pick: diffraction energy collapses best at the right curvature, flat events are unchanged.

Units: tau in samples, u in traces. With g = dx/dz (samples per trace) the coordinates
(tau, x' = g u) have unit exploding-reflector speed: a point at h gives tau = sqrt(h^2 + x'^2).
Stolt mapping: omega = sign(kz) sqrt(kz^2 + kx^2), kx = 2 pi f_u / g, amplitude factor |kz|/|omega|.
Zero: the record is shifted so that tau_z maps to depth 0 (tau_z in {0, -40, +17} samples).

Per window (D1a grid: centre every 1,000 traces, +-150 traces, outside pipeline mark +-400):
stored amplitude - moving median (101 traces), band-pass periods 8-40 samples, Tukey taper along
traces; migrate for each g on a log grid 0.5-5 (dz 0.06-0.006 m at dx 0.03); focus = varimax of
the migrated envelope in samples 30-250: sum(E^4) / (sum(E^2))^2 * N. g_hat = argmax (parabolic in
log g). Windows whose argmax sits on a grid end are flagged. Self-check on synthetic point
diffractions (g 1.0 / 2.5 / 4.0) first. Seed 20261126 (synthetic noise only).
Output: analysis/processing/m36_gprsd_migration_focus.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

from m36_gprsd_common import (BG_WINDOW, DX_M, ROADS, SEED, bandpass, excluded_mask,  # noqa: E402
                              line_ids, load_line, pipe_marks, qstats, road_bootstrap, road_of,
                              write_json)

HALF, STEP = 150, 1000
G_GRID = np.exp(np.linspace(np.log(0.5), np.log(5.0), 25))
TAUZ_LIST = (0.0, -40.0, 17.0)
Z0, Z1 = 30, 250


def stolt(d: np.ndarray, g: float, tau_z: float) -> np.ndarray:
    """Constant-speed Stolt migration of d[tau, u]; returns the migrated section [depth, u]."""
    n_t, n_u = d.shape
    shift = int(round(tau_z))
    pad_top = max(0, -shift)
    x = d[max(0, shift):, :] if shift >= 0 else np.vstack([np.zeros((pad_top, n_u)), d])
    nt = 1 << int(np.ceil(np.log2(1.5 * x.shape[0])))
    nu = 1 << int(np.ceil(np.log2(1.5 * n_u)))
    D = np.fft.fft2(x, s=(nt, nu))
    w = 2 * np.pi * np.fft.fftfreq(nt)            # rad / sample
    kx = 2 * np.pi * np.fft.fftfreq(nu) / g        # rad / x' unit
    kz = w                                         # output depth wavenumber grid
    out = np.zeros_like(D)
    order = np.argsort(w)
    ws = w[order]
    for j in range(nu):
        col = D[order, j]
        om = np.sign(kz) * np.sqrt(kz ** 2 + kx[j] ** 2)
        ok = np.abs(om) <= np.abs(ws).max()
        re = np.interp(om[ok], ws, col.real)
        im = np.interp(om[ok], ws, col.imag)
        fac = np.abs(kz[ok]) / np.maximum(np.abs(om[ok]), 1e-12)
        out[ok, j] = (re + 1j * im) * fac
    m = np.real(np.fft.ifft2(out))[: x.shape[0], :n_u]
    if shift >= 0:
        m = np.vstack([np.zeros((shift, n_u)), m])[:n_t]
    else:
        m = m[pad_top:pad_top + n_t]
    return m


def focus(m: np.ndarray) -> float:
    e = np.abs(signal.hilbert(m, axis=0))[Z0:Z1]
    e2 = (e ** 2).ravel()
    return float((e2 ** 2).sum() / (e2.sum() ** 2) * e2.size)


def scan(d: np.ndarray, tau_z: float) -> tuple[float, bool, list]:
    f = np.array([focus(stolt(d, g, tau_z)) for g in G_GRID])
    j = int(np.argmax(f))
    edge = j in (0, len(G_GRID) - 1)
    lg = np.log(G_GRID)
    x = lg[j]
    if 0 < j < len(f) - 1:
        den = f[j - 1] - 2 * f[j] + f[j + 1]
        if den < 0:
            x = lg[j] + 0.5 * (f[j - 1] - f[j + 1]) / den * (lg[1] - lg[0])
    return float(np.exp(x)), edge, f.tolist()


def prep(raw: np.ndarray) -> np.ndarray:
    taper = signal.windows.tukey(raw.shape[1], 0.2)[None, :]
    return bandpass(raw, 8.0, 40.0) * taper


def synthetic_check(rng) -> list:
    rows = []
    n_s, n_u = 321, 301
    k = np.arange(n_s)[:, None]
    u = np.arange(n_u)[None, :]
    for g in (1.0, 2.5, 4.0):
        img = np.zeros((n_s, n_u))
        for (u0, h) in ((60, 50.0), (150, 90.0), (240, 140.0), (110, 180.0)):
            tau = np.sqrt(h ** 2 + ((u - u0) * g) ** 2)
            x = (k - tau) / 20.0
            img += (h / tau) * (1 - 2 * (np.pi * x) ** 2) * np.exp(-(np.pi * x) ** 2)
        img += rng.normal(0, 0.2, img.shape)
        g_hat, edge, _ = scan(prep(img), 0.0)
        rows.append({"g_true": g, "g_hat": round(g_hat, 4), "rel_err": round(g_hat / g - 1, 4), "edge": edge})
    return rows


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    synth = synthetic_check(rng)
    print(json.dumps({"synthetic": synth}), flush=True)
    marks = pipe_marks()
    rows = []
    for fid in line_ids():
        a = load_line(fid)
        n = a.shape[1]
        ex = excluded_mask(fid, n, marks)
        for c in range(STEP, n - HALF, STEP):
            i0, i1 = c - HALF, c + HALF + 1
            if ex[i0:i1].any():
                continue
            m0, m1 = max(0, i0 - BG_WINDOW // 2), min(n, i1 + BG_WINDOW // 2)
            big = a[:, m0:m1]
            res = (big - ndimage.median_filter(big, size=(1, BG_WINDOW), mode="nearest"))[:, i0 - m0:i1 - m0]
            d = prep(res)
            row = {"file_id": fid, "road": road_of(fid), "centre_trace": c}
            for tz in TAUZ_LIST:
                g_hat, edge, curve = scan(d, tz)
                row[f"g_tz{int(tz)}"] = round(g_hat, 4)
                row[f"edge_tz{int(tz)}"] = edge
                row[f"focus_gain_tz{int(tz)}"] = round(max(curve) / min(curve), 4)
            rows.append(row)
        print(json.dumps({"line": fid, "windows": sum(r["file_id"] == fid for r in rows),
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    res = {}
    for tz in TAUZ_LIST:
        key = f"g_tz{int(tz)}"
        ok = [r for r in rows if not r[f"edge_tz{int(tz)}"]]
        dz = {rd: np.array([DX_M / r[key] for r in ok if r["road"] == rd]) for rd in ROADS}
        res[str(int(tz))] = {
            "n_windows": len(rows), "n_interior": len(ok),
            "g": qstats([r[key] for r in ok], 4),
            "dz_at_dx_0.03": qstats([DX_M / r[key] for r in ok], 5),
            "dz_by_road": {rd: qstats(dz[rd], 5) for rd in ROADS},
            "dz_bootstrap": road_bootstrap(dz),
            "focus_gain": qstats([r[f"focus_gain_tz{int(tz)}"] for r in rows], 4),
        }
    out = {"diagnostic": "D2m (report only, added after D2)", "g_grid": [float(G_GRID[0]), float(G_GRID[-1]),
                                                                          len(G_GRID)],
           "tau_z_list": TAUZ_LIST, "depth_window_samples": [Z0, Z1], "seed": SEED,
           "synthetic_check": synth, "result_by_tau_z": res, "windows": rows,
           "elapsed_s": round(time.time() - t0, 1)}
    p = write_json("m36_gprsd_migration_focus.json", out)
    print(json.dumps({"status": "ok", "out": str(p)}))


if __name__ == "__main__":
    main()
