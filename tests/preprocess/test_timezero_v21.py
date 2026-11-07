"""Air-coupled zero with two or more events: antenna height match (spec_depth v2.1).

Z8-Z13 of spec_depth_v1 section 11-1 (rule of section 4-2-1). Common setup:
500 MHz, dt 0.05 ns, eps_r 9, s 0.1 m, direct wave +3, surface reflection -2,
pipe 1.5 m deep (outside the zero window), SNR 20 dB, seed 0. A sub-surface
layer at depth ``z`` sits ``2 z sqrt(eps_r) / c`` after the surface reflection.
The depth error of a zero is ``-v (t_z - t_geom) / 2`` with ``t_geom`` the
geometric surface-reflection time. Measured values of these cases are in
``docs/processing/m29e2_two_event_zero_analytic.csv``.
"""

import math

import numpy as np
import pytest

from gpr_detection.io import fourtu
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT, load_dataset_paths
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS, Radargram
from gpr_detection.io.synthetic import load_synthetic
from gpr_detection.preprocess.filters import prepare_fitting_branch
from gpr_detection.preprocess.timezero import (
    AIR_MULTI_EVENT,
    AIR_SURFACE_NOT_FOUND,
    HEIGHT_MATCH_PERIODS,
    SEPARATION_PERIODS,
    SIGMA_ZERO_NS,
    ZeroEvents,
    ZeroRule,
    find_time_zero,
    select_zero,
    surface_event_index,
    surface_gap_ns,
)
from synthetic_bscan import T_SRC_NS, BScanSpec, cylinder_times, make_radargram

F_C_HZ = 500e6
PERIOD_NS = 1e9 / F_C_HZ
EPS_R = 9.0
OFFSET_M = 0.1
PIPE_DEPTH_M = 1.5
N_TRACES = 61
N_SAMPLES = 900
V_SOIL = SPEED_OF_LIGHT_M_PER_NS / math.sqrt(EPS_R)

DEPTH_TOLERANCE_M = 0.02
"""Height known: the zero must be within 2 cm of depth (spec Z9, Z10)."""

MERGED_BAND_HEIGHT_M = 0.05
"""Z8 (4TU-like B+L): direct coupling and surface reflection merge."""

BAND_LAYER_CASES = [
    (0.10, -0.6),
    (0.10, -1.6),
    (0.10, 0.6),
    (0.15, -0.6),
    (0.15, -1.6),
    (0.15, 0.6),
    (0.20, -1.6),
    (0.30, -1.6),
]
"""Z8: (layer depth z in m, layer amplitude) cases that give two events."""


def _surface_ns(height_m: float) -> float:
    """Geometric surface-reflection delay after the source, ns."""
    return 2.0 * math.hypot(height_m, OFFSET_M / 2.0) / SPEED_OF_LIGHT_M_PER_NS


def _spec(
    height_m: float,
    layers: list[tuple[float, float]],
    *,
    n_traces: int = N_TRACES,
    direct_amplitude: float = 3.0,
) -> BScanSpec:
    """Air-coupled spec with flat sub-surface layers ``[(z_m, amplitude)]``."""
    spec = BScanSpec(
        eps_r=EPS_R,
        f_c_hz=F_C_HZ,
        air_gap_m=height_m,
        offset_m=OFFSET_M,
        d_top_m=PIPE_DEPTH_M,
        n_traces=n_traces,
        n_samples=N_SAMPLES,
        direct_amplitude=direct_amplitude,
    )
    soil_ns = cylinder_times(spec.x_m, spec.x0_m, PIPE_DEPTH_M, V_SOIL, 0.0)
    spec.pipe_times_ns = soil_ns + 2.0 * height_m / SPEED_OF_LIGHT_M_PER_NS
    ones = np.ones(n_traces)
    for layer_m, amplitude in layers:
        layer_ns = _surface_ns(height_m) + 2.0 * layer_m * math.sqrt(EPS_R) / (
            SPEED_OF_LIGHT_M_PER_NS
        )
        spec.extra_events.append((layer_ns * ones, amplitude))
    return spec


def _zero_of(radargram: Radargram):
    return find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)


def _depth_error_m(zero, height_m: float) -> float:
    """``-v (t_z - t_geom) / 2``: positive when the zero makes depths too deep."""
    geometric_ns = T_SRC_NS + _surface_ns(height_m)
    return -V_SOIL * (zero.t_zero_ns - geometric_ns) / 2.0


def _assert_first_event_zero(zero) -> None:
    """``air_first_event`` with the merged-band contract (section 4-2-1 step 3)."""
    assert zero.zero_rule == ZeroRule.AIR_FIRST_EVENT
    assert zero.zero_event_label == "merged"
    assert zero.t_zero_ns == zero.events_ns[0]
    assert AIR_MULTI_EVENT in zero.flags
    assert zero.h_est_m is None
    assert zero.sigma_zero_ns == 0.25


# ------------------------------------------------------------------ Z8


@pytest.mark.parametrize("height_m", [MERGED_BAND_HEIGHT_M, None])
@pytest.mark.parametrize(("layer_m", "amplitude"), BAND_LAYER_CASES)
def test_z8_merged_band_and_layer_take_the_first_event(
    layer_m: float, amplitude: float, height_m: float | None
) -> None:
    """B+L (4TU-like): the later event is sub-surface; the old rule was 10-30 cm."""
    spec = _spec(MERGED_BAND_HEIGHT_M, [(layer_m, amplitude)])
    zero = _zero_of(make_radargram(spec, antenna_height_m=height_m))

    assert len(zero.events_ns) == 2
    _assert_first_event_zero(zero)
    assert AIR_SURFACE_NOT_FOUND not in zero.flags
    assert 0.0 < _depth_error_m(zero, MERGED_BAND_HEIGHT_M) <= DEPTH_TOLERANCE_M


def test_z8_first_event_rule_uses_the_single_event_uncertainty() -> None:
    assert SIGMA_ZERO_NS[ZeroRule.AIR_FIRST_EVENT] == SIGMA_ZERO_NS["air_single_event"]
    spec = _spec(MERGED_BAND_HEIGHT_M, [(0.10, -0.6)])
    zero = _zero_of(make_radargram(spec, antenna_height_m=None))
    assert zero.bias_notes[0].startswith("air_single_event: merged peak")
    assert zero.bias_notes[1].startswith("air_first_event: later events")
    assert zero.bias_notes[1].endswith("(antenna height unknown)")


# ------------------------------------------------------------------ Z9


@pytest.mark.parametrize("height_m", [0.3, 0.4, 0.5])
def test_z9_known_height_picks_the_separated_surface_reflection(
    height_m: float,
) -> None:
    zero = _zero_of(make_radargram(_spec(height_m, [])))

    assert zero.zero_rule == ZeroRule.AIR_SURFACE_LATE
    assert zero.zero_event_label == "surface_reflection"
    assert len(zero.events_ns) == 2
    assert abs(_depth_error_m(zero, height_m)) <= DEPTH_TOLERANCE_M
    assert abs(zero.h_est_m - height_m) <= 0.06
    assert zero.sigma_zero_ns == 0.3
    assert AIR_MULTI_EVENT not in zero.flags


@pytest.mark.parametrize("height_m", [0.3, 0.4, 0.5])
def test_z9_unknown_height_takes_the_direct_coupling_and_is_deep(
    height_m: float,
) -> None:
    """Documented bias of an unknown height: about h / sqrt(eps_r) too deep."""
    zero = _zero_of(make_radargram(_spec(height_m, []), antenna_height_m=None))

    _assert_first_event_zero(zero)
    bound_m = height_m / math.sqrt(EPS_R) + DEPTH_TOLERANCE_M
    assert 0.0 < _depth_error_m(zero, height_m) <= bound_m


# ------------------------------------------------------------------ Z10


@pytest.mark.parametrize(("layer_m", "amplitude"), [(0.08, 1.6), (0.11, -1.6)])
def test_z10_surface_is_the_height_matched_middle_of_three_events(
    layer_m: float, amplitude: float
) -> None:
    height_m = 0.3
    spec = _spec(height_m, [(layer_m, amplitude)])
    first_pass = _zero_of(make_radargram(spec))
    assert len(first_pass.events_ns) == 3
    candidates = {
        "on_surface": first_pass.events_ns[1],
        "on_layer": first_pass.events_ns[2],
    }
    zero = _zero_of(make_radargram(spec, t0_candidates_ns=candidates))

    assert zero.zero_rule == ZeroRule.AIR_SURFACE_LATE
    assert zero.t_zero_ns == zero.events_ns[1]
    assert abs(_depth_error_m(zero, height_m)) <= DEPTH_TOLERANCE_M
    assert 0.30 <= zero.h_est_m <= 0.36
    assert zero.meta_t0["on_surface"][1] == "surface_reflection"
    assert zero.meta_t0["on_layer"][1] == "unclassified"


# ------------------------------------------------------------------ Z11


def test_z11a_known_height_without_a_matching_gap_falls_back_to_the_first() -> None:
    """Delta_h 3.68 ns vs gap 2.27 ns: off by 1.41 ns > tau_h 1.0 ns."""
    spec = _spec(MERGED_BAND_HEIGHT_M, [(0.10, -0.6)])
    zero = _zero_of(make_radargram(spec, antenna_height_m=0.6))

    _assert_first_event_zero(zero)
    assert AIR_SURFACE_NOT_FOUND in zero.flags
    # Section 4-2-1 implementation note: the height is known here.
    assert zero.bias_notes[1].endswith("(antenna height known, no event gap matched)")


def test_z11b_low_known_height_merges_the_surface_into_the_first_event() -> None:
    """Delta_h 0.41 ns < Delta_sep 1.5 ns: step 2, no search for a surface."""
    spec = _spec(MERGED_BAND_HEIGHT_M, [(0.10, -0.6)])
    zero = _zero_of(make_radargram(spec, antenna_height_m=0.1))

    _assert_first_event_zero(zero)
    assert AIR_SURFACE_NOT_FOUND not in zero.flags


GAP_H_NS = 2.0
"""Z11(c): expected gap; with a 2 ns period tau_h = 1.0 ns, Delta_sep = 1.5 ns.
Values are exact in binary so ties and boundaries are real."""


@pytest.mark.parametrize(
    ("gaps_ns", "gap_h_ns", "expected"),
    [
        ([GAP_H_NS + 1.0], GAP_H_NS, 0),  # boundary is included
        ([GAP_H_NS + 1.0 + 1e-6], GAP_H_NS, None),
        ([GAP_H_NS - 0.25, GAP_H_NS + 0.25], GAP_H_NS, 0),  # tie -> earlier
        ([GAP_H_NS + 0.75, GAP_H_NS - 0.5], GAP_H_NS, 1),  # closest, not first
        ([1.4], 1.4, None),  # Delta_h < Delta_sep: merged, no match
        ([], GAP_H_NS, None),
    ],
)
def test_z11c_surface_event_index(
    gaps_ns: list[float], gap_h_ns: float, expected: int | None
) -> None:
    assert surface_event_index(gaps_ns, gap_h_ns, PERIOD_NS) == expected


def test_z11c_constants_and_height_gap() -> None:
    assert HEIGHT_MATCH_PERIODS == 0.5
    assert SEPARATION_PERIODS == 0.75
    expected_ns = (2.0 * math.hypot(0.3, 0.05) - 0.1) / SPEED_OF_LIGHT_M_PER_NS
    assert surface_gap_ns(0.3, 0.1) == pytest.approx(expected_ns)
    assert surface_gap_ns(0.3, None) == pytest.approx(0.6 / SPEED_OF_LIGHT_M_PER_NS)


# ------------------------------------------------------------------ Z11(d), Z11(e)
# Judge mutants JC (height match ignores the Tx-Rx offset s) and JE (no-scale
# multi-event input takes the old last-event rule), spec section 11-1 v2.1.1.

FAKE_FIRST_EVENT_K = 100.0
FAKE_DT_NS = 0.05


def _fake_events(events_k: tuple[float, ...], dt_ns: float | None) -> ZeroEvents:
    """Early events as :func:`detect_zero_events` would report them (no windows)."""
    return ZeroEvents(
        events_k=events_k,
        amplitudes=tuple(1.0 for _ in events_k),
        dt_ns=dt_ns,
        f_c_hz=F_C_HZ,
        pick_mode="envelope",
        window_end_k=int(max(events_k)) + 50,
        h_est_m=None,
        window_events_k=(),
        excluded_traces=0,
        flags=(),
    )


def _air_radargram(height_m: float | None, *, has_scale: bool = True) -> Radargram:
    return _minimal_radargram(
        dt_ns=FAKE_DT_NS if has_scale else None,
        dx_m=0.02 if has_scale else None,
        f_c_hz=F_C_HZ,
        antenna_offset_m=OFFSET_M,
        antenna_height_m=height_m,
    )


@pytest.mark.parametrize("height_m", [0.3, 0.4, 0.5])
def test_z11d_ignoring_the_offset_shifts_delta_h_by_less_than_tau_h(
    height_m: float,
) -> None:
    """Why JC survives on real data: s 0.1 m moves Delta_h by ~0.31 ns < 1.0 ns.

    Section 4-2-1 tau_h basis (ii) sizes tau_h to absorb this difference, so a
    resolvable surface event matches with or without s.
    """
    shift_ns = surface_gap_ns(height_m, None) - surface_gap_ns(height_m, OFFSET_M)
    assert 0.30 <= shift_ns <= 0.32
    assert shift_ns <= HEIGHT_MATCH_PERIODS * PERIOD_NS


def test_z11d_height_match_uses_the_offset() -> None:
    """Catches JC: a gap that matches Delta_h(s) but not Delta_h(s = 0).

    Ignoring s raises Delta_h (h 0.3 m: 1.695 -> 2.001 ns), so the gaps that
    only the offset-aware rule accepts lie in [Delta_h(s) - tau_h,
    Delta_h(0) - tau_h). The gap sits in the middle of that 0.31 ns band. It is
    shorter than Delta_sep, so no real radargram resolves it; this checks only
    the gap arithmetic of the height match, through ``select_zero``.
    """
    height_m = 0.3
    tau_h_ns = HEIGHT_MATCH_PERIODS * PERIOD_NS
    gap_with_offset_ns = surface_gap_ns(height_m, OFFSET_M)
    gap_without_offset_ns = surface_gap_ns(height_m, None)
    gap_ns = (gap_with_offset_ns + gap_without_offset_ns) / 2.0 - tau_h_ns
    assert abs(gap_ns - gap_with_offset_ns) < tau_h_ns
    assert abs(gap_ns - gap_without_offset_ns) > tau_h_ns

    surface_k = FAKE_FIRST_EVENT_K + gap_ns / FAKE_DT_NS
    events = _fake_events((FAKE_FIRST_EVENT_K, surface_k), FAKE_DT_NS)
    zero = select_zero(_air_radargram(height_m), events)

    assert zero.zero_rule == ZeroRule.AIR_SURFACE_LATE
    assert zero.k_zero == surface_k
    assert AIR_SURFACE_NOT_FOUND not in zero.flags


@pytest.mark.parametrize("height_m", [0.3, None])
def test_z11e_no_scale_multi_event_takes_the_first_event(
    height_m: float | None,
) -> None:
    """Catches JE: without dt the gaps have no ns, so step 1 cannot run (step 3).

    The second event sits at the 0.3 m surface gap in samples of 0.05 ns, so
    the old last-event rule would pick it.
    """
    surface_k = FAKE_FIRST_EVENT_K + surface_gap_ns(0.3, OFFSET_M) / FAKE_DT_NS
    events = _fake_events((FAKE_FIRST_EVENT_K, surface_k), None)
    zero = select_zero(_air_radargram(height_m, has_scale=False), events)

    assert zero.zero_rule == ZeroRule.AIR_FIRST_EVENT
    assert zero.zero_event_label == "merged"
    assert zero.k_zero == FAKE_FIRST_EVENT_K
    assert zero.t_zero_ns is None
    assert zero.h_est_m is None
    assert AIR_MULTI_EVENT in zero.flags
    assert "zero_px_unverified" in zero.flags
    assert AIR_SURFACE_NOT_FOUND not in zero.flags


# ------------------------------------------------------------------ Z12


def test_z12a_first_event_meta_labels_are_merged_or_unclassified() -> None:
    spec = _spec(MERGED_BAND_HEIGHT_M, [(0.10, -0.6)])
    first_pass = _zero_of(make_radargram(spec))
    candidates = {
        "on_lobe": first_pass.events_ns[0] + 0.78,
        "on_layer": first_pass.events_ns[1],
    }
    zero = _zero_of(make_radargram(spec, t0_candidates_ns=candidates))

    assert zero.zero_rule == ZeroRule.AIR_FIRST_EVENT
    assert zero.meta_t0["on_lobe"][1] == "merged"
    assert zero.meta_t0["on_layer"][1] == "unclassified"


def test_z12b_window_qc_tracks_the_first_event() -> None:
    """A layer on part of the line only (4TU pattern): QC follows the first event."""
    n_traces = 700
    spec = _spec(MERGED_BAND_HEIGHT_M, [], n_traces=n_traces)
    layer_ns = _surface_ns(MERGED_BAND_HEIGHT_M) + 2.0 * 0.10 * math.sqrt(EPS_R) / (
        SPEED_OF_LIGHT_M_PER_NS
    )
    layer_amplitude = np.zeros(n_traces)
    layer_amplitude[: n_traces // 2] = -1.0
    spec.extra_events = [(np.full(n_traces, layer_ns), layer_amplitude)]
    zero = _zero_of(make_radargram(spec, antenna_height_m=None))

    assert len(zero.events_ns) == 2
    assert zero.zero_rule == ZeroRule.AIR_FIRST_EVENT
    assert zero.window_std_ns is not None
    assert zero.window_std_ns < 0.05
    assert "zero_window_missing" not in zero.flags
    assert "zero_window_event_count_mismatch" in zero.flags


# ------------------------------------------------------------------ Z13


def _minimal_radargram(**overrides) -> Radargram:
    fields = {
        "data": np.zeros((4, 3), dtype=np.float32),
        "dt_ns": 0.1,
        "dx_m": 0.02,
        "air_gap": "air",
        "amplitude_processed": False,
        "dataset": "unit",
        "line_id": "line",
    }
    fields.update(overrides)
    return Radargram(**fields)


@pytest.mark.parametrize("height_m", [0.0, -0.1, math.nan, math.inf])
def test_z13_radargram_rejects_a_non_positive_or_non_finite_height(
    height_m: float,
) -> None:
    with pytest.raises(ValueError, match="antenna_height_m"):
        _minimal_radargram(antenna_height_m=height_m)


def test_z13_radargram_height_defaults_to_unknown() -> None:
    assert _minimal_radargram().antenna_height_m is None
    assert _minimal_radargram(antenna_height_m=0.3).antenna_height_m == 0.3


def test_z13_synthetic_loader_passes_the_air_gap_as_height() -> None:
    if not (DEFAULT_SYNTHETIC_ROOT / "manifest.csv").is_file():
        pytest.skip(f"synthetic data not found at {DEFAULT_SYNTHETIC_ROOT}")
    assert load_synthetic("M20").antenna_height_m == 0.3
    assert load_synthetic("M02").antenna_height_m is None
    assert load_synthetic("M20", include_height=False).antenna_height_m is None


@pytest.mark.data
def test_z13_fourtu_loader_leaves_the_height_unknown() -> None:
    root = load_dataset_paths().fourtu
    if not root.is_dir():
        pytest.skip(f"4TU data not found at {root}")
    activity_id, path = fourtu.list_fourtu_lines(root)[0]
    radargram = fourtu.load_fourtu_radargram(root, activity_id, path)
    assert radargram.air_gap == "air"
    assert radargram.antenna_height_m is None
