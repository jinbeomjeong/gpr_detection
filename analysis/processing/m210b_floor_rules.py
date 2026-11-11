"""M2-10 (b) detection-normalisation floor rules of the apex-candidate selector (prototype).

Reference prototype for docs/processing/m210b_selector_floor.md. Pure functions,
no file I/O. The ``current`` rule reproduces step 3 of
``fourtu_apex_candidate_select.apex_candidates_for_radargram`` exactly (the
detection normalisation, lines 279-283 at 90845b4 and 1bba359); the other rules are the
improvement candidates of the document.

Notation (document section 2):
    S      background-removed amplitudes, [n_samples, n_traces]
    E      envelope |hilbert(S)| (``pick_signal(..., "envelope")``)
    k_lo   ceil((t_z + 1/f_c) / dt) + 2      first searched row (mute)
    k_hi   n_samples - 16                    first row after the search region
    m(k)   median_j E(k, j)                  per-row median over traces
    l(k)   uniform_filter1d(m, w), w = max(3, int(1/(f_c dt)))
    F      floor; lambda(k) = max(l(k), F); D = E / lambda; peak iff local max
           (min_distance 8) and D >= SNR_MIN, at most NUM_PEAKS strongest.

Rules for F:
    current   0.05 * max_k l(k)                              (selector v1)
    A_skip1   0.05 * max_{k >= k_lo + n1} l(k), n1 = ceil(1/(f_c dt))
    B_noise   max(K_NOISE * sigma_n / SNR_MIN, eps32 * max E_sub),
              sigma_n = 1.4826 * MAD(S[k_lo:k_hi, :])        (spec_depth 5-7 (i))
    B_full    as B_noise, sigma_n = 1.4826 * MAD(S[0:k_hi, :]) (whole record above
              the end trim: independent of t_z, because S does not depend on t_z)
    B_min     as B_noise, sigma_n = min_k smooth_w(1.4826 * MAD_j S(k, j)) over the
              searched rows (quietest row = instrument noise; an earlier zero adds rows,
              which can only lower the minimum, so the floor never rises when t_z moves)
    C_none    eps32 * max E_sub (numerical guard only)
"""

import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.feature import peak_local_max

SNR_MIN = 2.0  # fourtu_apex_candidate_select.SNR_MIN
FLOOR_REL = 0.05  # fourtu_apex_candidate_select line 282 (90845b4)
NUM_PEAKS = 300  # fourtu_apex_candidate_select.MAX_CANDIDATES
MIN_DISTANCE = 8  # peak_local_max(min_distance=8)
RECORD_END_TRIM = 16  # k_hi = n_samples - 16
K_NOISE = 5.0  # spec_depth_v1 section 5-7 item 1 (i), apex-hint event floor
MAD_SCALE = 1.4826
EPS32 = float(np.finfo(np.float32).eps)

RULES = ("current", "A_skip1", "B_noise", "B_full", "B_min", "C_none")


def search_rows(t_z_ns: float, f_c_hz: float, dt_ns: float, n_samples: int) -> tuple[int, int]:
    """(k_lo, k_hi) of the selector search region."""
    k_lo = math.ceil((t_z_ns + 1e9 / f_c_hz) / dt_ns) + 2
    return k_lo, n_samples - RECORD_END_TRIM


def period_rows(f_c_hz: float, dt_ns: float) -> int:
    return max(3, int(1e9 / f_c_hz / dt_ns))


def level_profile(e_sub: np.ndarray, f_c_hz: float, dt_ns: float) -> np.ndarray:
    """Smoothed per-row median l(k) of the searched envelope rows."""
    return uniform_filter1d(np.median(e_sub, axis=1), size=period_rows(f_c_hz, dt_ns))


def noise_sigma(s_sub: np.ndarray) -> float:
    """1.4826 * MAD of the background-removed amplitudes in the searched rows."""
    a = np.asarray(s_sub, dtype=np.float64)
    return float(MAD_SCALE * np.median(np.abs(a - np.median(a))))


def row_noise_sigma_min(s_sub: np.ndarray, f_c_hz: float, dt_ns: float) -> float:
    """Minimum over rows of the one-period smoothed row-wise 1.4826 * MAD over traces."""
    a = np.asarray(s_sub, dtype=np.float64)
    med = np.median(a, axis=1, keepdims=True)
    row = MAD_SCALE * np.median(np.abs(a - med), axis=1)
    return float(uniform_filter1d(row, size=period_rows(f_c_hz, dt_ns)).min())


def floor_value(
    rule: str,
    level: np.ndarray,
    e_sub: np.ndarray,
    s_sub: np.ndarray | None,
    f_c_hz: float,
    dt_ns: float,
    s_full: np.ndarray | None = None,
) -> float:
    guard = EPS32 * float(e_sub.max()) if e_sub.size else 0.0
    if rule == "current":
        return FLOOR_REL * float(level.max())
    if rule == "A_skip1":
        n1 = math.ceil(1e9 / f_c_hz / dt_ns)
        tail = level[n1:] if level.size > n1 else level[-1:]
        return FLOOR_REL * float(tail.max())
    if rule == "B_noise":
        if s_sub is None:
            raise ValueError("B_noise needs the background-removed amplitudes")
        return max(K_NOISE * noise_sigma(s_sub) / SNR_MIN, guard)
    if rule == "B_full":
        if s_full is None:
            raise ValueError("B_full needs the whole-record amplitudes")
        return max(K_NOISE * noise_sigma(s_full) / SNR_MIN, guard)
    if rule == "B_min":
        if s_sub is None:
            raise ValueError("B_min needs the background-removed amplitudes")
        return max(K_NOISE * row_noise_sigma_min(s_sub, f_c_hz, dt_ns) / SNR_MIN, guard)
    if rule == "C_none":
        return max(guard, np.finfo(np.float64).tiny)
    raise ValueError(f"unknown rule {rule}")


def detect(
    E: np.ndarray,
    S: np.ndarray | None,
    k_lo: int,
    k_hi: int,
    f_c_hz: float,
    dt_ns: float,
    edge_traces: int,
    rule: str = "current",
    num_peaks: float = NUM_PEAKS,
) -> tuple[list[tuple[int, int, float]], dict, np.ndarray, np.ndarray]:
    """Detection peaks ``(k, j, D)`` (k absolute row), diagnostics, l(k) and lambda(k).

    ``num_peaks`` = NUM_PEAKS reproduces the selector; ``np.inf`` removes the cap.
    The edge rule is applied after ``peak_local_max`` (as in the selector).
    """
    e_sub = E[k_lo:k_hi]
    s_sub = None if S is None else S[k_lo:k_hi]
    level = level_profile(e_sub, f_c_hz, dt_ns)
    s_full = None if S is None else S[:k_hi]
    F = floor_value(rule, level, e_sub, s_sub, f_c_hz, dt_ns, s_full)
    lam = np.maximum(level, F)
    detect_map = e_sub / lam[:, None]
    kw = {} if not np.isfinite(num_peaks) else {"num_peaks": int(num_peaks)}
    pk = peak_local_max(
        detect_map,
        min_distance=MIN_DISTANCE,
        threshold_abs=SNR_MIN,
        exclude_border=False,
        **kw,
    )
    n_traces = E.shape[1]
    peaks = [
        (int(k) + k_lo, int(j), float(detect_map[k, j]))
        for k, j in pk
        if edge_traces <= j < n_traces - edge_traces
    ]
    diag = {
        "L": float(level.max()),
        "k_argmax_level": int(np.argmax(level)) + k_lo,
        "F": float(F),
        "frac_rows_floor_active": float(np.mean(level < F)),
        "n_raw_peaks": int(len(pk)),
        "cap_hit": bool(np.isfinite(num_peaks) and len(pk) >= num_peaks),
    }
    return peaks, diag, level, lam
