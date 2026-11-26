"""Synthetic schematic of the M3-4 GPR-SD box convention (no real data; safe for the public repo).

Renders synthetic hyperbolas with a ringing wavelet in the M3-4 label-view style (symmetric p99 clip, 8-bit grey,
2 px/trace, 2 px/sample) and draws the convention:
 box top = apex lobe peak - 0.75 P, bottom = apex peak + min(6 P, last visible ring), sides where the
 apex arc has descended 3 P, apex point = peak of the first strong lobe at the apex trace.
Panel (a) single hyperbola, panel (b) a near pair (separate boxes) and an arc whose apex lies outside the
window (ignore). P = 20.0 samples (M3-6 closed: reflected carrier 19.3-22.3 samples, M3-3b fixed P 20.0;
docs/processing/m36_gprsd_scale.md s13-s14). A weak flat, trace-independent component with period 14.6 samples
is added to mimic the time-fixed equipment stripes (EM E2): it does not follow the arcs and is NOT P.
r1 (2026-11-26): P 14.6 -> 20.0, equipment stripes added.
Output: docs/experiments/figures/m34_box_convention_schematic.png
Example: conda run -n gpr-dev python -I analysis/detection/m34_box_schematic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "experiments" / "figures" / "m34_box_convention_schematic.png"
P = 20.0
P_EQUIP = 14.6  # time-fixed equipment stripe period (not the carrier)
NS, NT = 260, 240
SEED = 0


def wavelet(t):
    # ringing wavelet: onset at t = -0.75 P, envelope peak at 0, decays over ~6 P
    env = np.where(t < 0, np.exp(-(t / (0.45 * P)) ** 2), np.exp(-t / (1.8 * P)))
    return env * np.cos(2 * np.pi * t / P)


def add_hyp(img, u0, ta, k, amp=1.0):
    u = np.arange(NT)
    tau = np.sqrt(ta ** 2 + ((u - u0) * k) ** 2)
    s = np.arange(NS)[:, None]
    img += amp * wavelet(s - tau[None, :]) * (1.0 / (1.0 + ((u - u0) / 60.0) ** 2))[None, :]
    return tau


def add_equipment(img, amp=0.12):
    # flat stripes fixed in time (same sample on every trace), period P_EQUIP
    s = np.arange(NS)[:, None]
    img += amp * np.cos(2 * np.pi * s / P_EQUIP) * np.ones((1, NT))


def view(img):
    c = np.percentile(np.abs(img), 99)
    v = np.clip(img / c, -1, 1)
    g = ((v + 1) * 127.5).astype(np.uint8)
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)


def box_for(tau, u0, ta, k):
    top = ta - 0.75 * P
    bottom = ta + 6 * P
    du = np.sqrt((ta + 3 * P) ** 2 - ta ** 2) / k
    return u0 - du, u0 + du, top, bottom


def main() -> int:
    import cv2
    rng = np.random.default_rng(SEED)
    panels = []
    # (a) single
    a = rng.normal(0, 0.05, (NS, NT))
    for r in (20, 34):  # flat layers
        a += 0.6 * wavelet(np.arange(NS)[:, None] - r) * np.ones((1, NT))
    add_equipment(a)
    k = 2.2
    tau = add_hyp(a, 120, 90, k)
    va = cv2.cvtColor(view(a), cv2.COLOR_GRAY2BGR)
    x0, x1, y0, y1 = box_for(tau, 120, 90, k)
    cv2.rectangle(va, (int(2 * x0), int(2 * y0)), (int(2 * x1), int(2 * y1)), (0, 200, 0), 2)
    cv2.circle(va, (240, 180), 5, (0, 0, 255), -1)
    for i_, lab in enumerate(("green box: top = apex peak - 0.75P (P=20 -> 30 px)",
                              "red dot: apex = peak of 1st strong lobe",
                              "bottom = last visible curved ring (<= apex + 6P = 240 px)",
                              "sides: apex arc has descended ~3P (120 px)",
                              "flat stripes (14.6 smp) = equipment, not P")):
        cv2.putText(va, lab, (8, 2 * NS - 82 + 16 * i_), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
    cv2.putText(va, "(a)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(va)
    # (b) near pair + apex outside window (right edge)
    b = rng.normal(0, 0.05, (NS, NT))
    for r in (20, 34):
        b += 0.6 * wavelet(np.arange(NS)[:, None] - r) * np.ones((1, NT))
    add_equipment(b)
    t1 = add_hyp(b, 70, 80, k)
    t2 = add_hyp(b, 125, 95, k)
    add_hyp(b, 275, 70, k, amp=0.8)  # apex outside the window
    vb = cv2.cvtColor(view(b), cv2.COLOR_GRAY2BGR)
    for (u0, ta, tt) in ((70, 80, t1), (125, 95, t2)):
        x0, x1, y0, y1 = box_for(tt, u0, ta, k)
        cv2.rectangle(vb, (int(2 * x0), int(2 * y0)), (int(2 * x1), int(2 * y1)), (0, 200, 0), 2)
        cv2.circle(vb, (2 * u0, 2 * ta), 5, (0, 0, 255), -1)
    cv2.rectangle(vb, (2 * 200, 2 * 50), (2 * NT - 2, 2 * 200), (0, 140, 255), 2)
    cv2.putText(vb, "ignore: apex outside window", (2 * 150, 2 * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 140, 255), 1)
    cv2.putText(vb, "(b) two apexes -> two boxes (overlap allowed)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    panels.append(vb)
    pad = np.full((panels[0].shape[0], 12, 3), 40, np.uint8)
    out = np.hstack([panels[0], pad, panels[1]])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), out)
    print(json.dumps({"status": "ok", "out": str(OUT), "shape": list(out.shape)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
