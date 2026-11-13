"""Expected values for spec_detection_image v0.2 tests (line_scale, tile_layout)."""
import math

C = 0.299792458


def line_scale(n_s, n_tr, dx, period, dps, p_star, tile=256):
    s_y = period / p_star
    s_x = s_y * dps / dx
    h_nom = n_s / s_y
    compressed = round(h_nom) > tile
    if compressed:
        f = (n_s / tile) / s_y
        s_y *= f
        s_x *= f
    h_c = max(1, round(n_s / s_y))
    w1 = max(1, round(n_tr / s_x))
    tpp = n_tr / w1
    spp = n_s / h_c
    o_phys = max(64, math.ceil(2.0 / (tpp * dx) - 1e-9))
    return dict(s_y=round(s_y, 4), s_x=round(s_x, 4), H_c=h_c, W1=w1, tpp=round(tpp, 5), spp=round(spp, 5),
                compressed=compressed, o_phys=o_phys)


def n_tiles(w1, o, tile=256):
    if w1 <= tile:
        return 1
    s = tile - o
    return 1 + math.ceil((w1 - tile) / s)


v9 = C / 3.0
cases = {
    "4TU 01.1/Path1 eps9 dt0.097 P2.51ns": (512, 304, 0.02, 2.51 / 0.097, 0.097 * v9 / 2, 4.06),
    "4TU 09.1/Path2 eps? dt0.195 (eps9)": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * v9 / 2, 4.06),
    "4TU dt0.195 P*8 (compression)": (512, 12186, 0.02, 2.51 / 0.195, 0.195 * v9 / 2, 8.0),
    "GZ 30ns 710 P2.93 v0.10": (512, 710, 0.024, 2.93 / (30 / 512), (30 / 512) * 0.10 / 2, 4.06),
    "GPR-SD CementRoad2_1 P51.2 dps0.012": (321, 9411, 0.03, 51.2, 0.012, 4.06),
    "4TU 03.9/Path5 9 traces": (512, 9, 0.02, 2.51 / 0.098, 0.098 * v9 / 2, 4.06),
}
for name, args in cases.items():
    r = line_scale(*args)
    r["tiles_O128"] = n_tiles(r["W1"], max(128, r["o_phys"]))
    r["tiles_O192"] = n_tiles(r["W1"], max(192, r["o_phys"]))
    print(name, r)

for w1 in (1, 97, 255, 256, 257, 300, 383, 512, 2250, 14516, 27628):
    print(w1, [n_tiles(w1, o) for o in (0, 64, 128, 141, 192)])


def o_phys_capped(m_per_px, p_star, w_periods=20.0, o_min=64, a_max=1.0):
    """v0.3: min(2 a_max, w_periods * lambda/2) in L1 px; lambda/2 = P* px in iso px."""
    return max(o_min, math.ceil(min(2 * a_max / m_per_px, w_periods * p_star) - 1e-9))


for name, args in cases.items():
    r = line_scale(*args)
    mpp = r["tpp"] * args[2]
    print("o_phys v0.3", name, o_phys_capped(mpp, args[5]))
# Guangzhou 2000 MHz example: dt 0.039 ns, dx 0.01 m, P 0.5 ns, v 0.10
s_y = (0.5 / 0.039) / 4.06
mpp = s_y * (0.10 * 0.039 / 2)
print("2000MHz m/px", round(mpp, 5), "2a_max px", math.ceil(2 / mpp), "o_phys v0.3", o_phys_capped(mpp, 4.06))

# box-average response at 1.5x new Nyquist
for s in (3, 6.3, 12.6):
    f = 0.75 / s
    print("alias", s, abs(math.sin(math.pi * f * s) / (s * math.sin(math.pi * f))))
