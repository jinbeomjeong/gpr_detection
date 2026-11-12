"""K7 generator variants and c diagnostics (m210_k7_generator.md v1.1 4-1, T1-T8).

T2 and T3 build analytic B-scans that reproduce the two defects with the current
generator before checking the fix; T6 compares the default path with the
generator as sealed at 752e783 (blob 7cc2bf2b, loaded from git), and a data
test regenerates two 4TU activities and compares them with the committed csvs.
"""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.signal import hilbert

from synthetic_bscan import (
    T_SRC_NS,
    BScanSpec,
    cylinder_times,
    make_bscan,
    make_radargram,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "processing"))

import fourtu_apex_candidate_select as generator  # noqa: E402

from gpr_detection.depth.config import DepthConfig  # noqa: E402
from gpr_detection.depth.estimate import prepare_depth_inputs  # noqa: E402

GENERATOR_PATH = "analysis/processing/fourtu_apex_candidate_select.py"
V1_COMMIT = "752e783"
V1_BLOB_PREFIX = "7cc2bf2b"
F_C_HZ = 500e6
PERIOD_NS = 1e9 / F_C_HZ
DT_NS = 0.05
APEX_TRACE = 80
WINDOW_TRACES = 3
A1 = {"start_rule": "envelope_max"}
A1B2 = {"start_rule": "envelope_max", "rank_rule": "snr"}
FOURTU_FILES = ("fourtu_apex_candidates_v1.csv", "fourtu_apex_candidate_lines_v1.csv")


# ----------------------------------------------------------------- fixtures


def _air_spec(n_samples: int) -> BScanSpec:
    """Air-coupled line with a clear zero (the (b) harness fake-line settings)."""
    return BScanSpec(
        f_c_hz=F_C_HZ, dt_ns=DT_NS, dx_m=0.02, n_traces=161, n_samples=n_samples,
        snr_db=None, air_gap_m=0.001, offset_m=0.0, direct_amplitude=30.0,
        surface_amplitude=-20.0, pipe_amplitude=1.0, eps_r=9.0,
    )  # fmt: skip


def _alternating(n_traces: int) -> np.ndarray:
    return np.where(np.arange(n_traces) % 2 == 0, 1.0, -1.0)


def t2_line():
    """T2: 0.6 m hyperbola + a flat band centred 1.0 ns after its apex.

    The band alternates its sign from trace to trace: a band of one sign would
    be removed by the 301-trace background removal, while this one keeps a flat
    envelope on every trace and so raises the row level ``ell`` (K7-a).
    """
    spec = _air_spec(600)
    spec.pipe_times_ns = cylinder_times(spec.x_m, spec.x0_m, 0.6, spec.velocity, 0.0)
    t_apex_ns = float(spec.pipe_times_ns[APEX_TRACE])
    band_times = np.full(spec.n_traces, t_apex_ns + 1.0)
    spec.extra_events = [(band_times, 0.1 * _alternating(spec.n_traces))]
    return make_radargram(spec, line_id="unit/T2"), T_SRC_NS + t_apex_ns


T3_PIPES = (65, 95)
T3_INNER_PICKS = 9
T3_WEAK_DEPTHS_M = (1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9)


def t3_line():
    """T3: two 0.6 m hyperbolas 0.6 m apart, inner wings cut after 9 traces, and
    seven weak deep hyperbolas with 50/50 wings at the middle trace (K7-b)."""
    spec = _air_spec(1000)
    spec.pipe_amplitude = 0.0
    spec.pipe_times_ns = np.zeros(spec.n_traces)
    traces = np.arange(spec.n_traces)
    events, apex_times = [], {}
    for apex, inner_side in ((T3_PIPES[0], +1), (T3_PIPES[1], -1)):
        times = cylinder_times(spec.x_m, spec.x_m[apex], 0.6, spec.velocity, 0.0)
        is_cut = inner_side * (traces - apex) > T3_INNER_PICKS
        events.append((times, np.where(is_cut, 0.0, 1.0)))
        apex_times[apex] = T_SRC_NS + float(times[apex])
    for depth_m in T3_WEAK_DEPTHS_M:
        times = cylinder_times(spec.x_m, spec.x_m[80], depth_m, spec.velocity, 0.0)
        events.append((times, 0.3))
    spec.extra_events = events
    return make_radargram(spec, line_id="unit/T3"), apex_times


def run(radargram, **variant):
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    return generator.apex_candidates_for_radargram(
        radargram, inputs, act="UNIT", path="line", **variant
    )


def in_window(item, apex_trace: int, t_apex_ns: float) -> bool:
    return (
        abs(item["trace"] - apex_trace) <= WINDOW_TRACES
        and abs(item["t_ns"] - t_apex_ns) <= 0.5 * PERIOD_NS
    )


def peak_of(line, item):
    """First accepted detection peak whose end point is ``item``."""
    return next(
        peak
        for peak in line["_peaks"]
        if peak["status"] == "accepted"
        and peak["trace"] == item["trace"]
        and round(peak["sample"], 2) == item["sample"]
    )


# ----------------------------------------------------------------- T1, T8


@pytest.mark.parametrize(
    ("column", "start", "expected"),
    [
        ([0, 1, 2, 3, 5, 4, 1], 1, 4),  # rising flank -> the local maximum
        ([0, 1, 2, 3, 5, 4, 1], 6, 4),  # falling flank, climbs earlier
        ([0, 1, 2, 3, 5, 4, 1], 4, 4),  # already on the maximum
        ([0, 1, 2, 3, 4], 1, 4),  # stops at the array end
        ([4, 3, 2], 0, 0),  # the first sample is a maximum
        ([1, 2, 2, 1], 1, 1),  # flat top: no strictly larger neighbour
        ([1, 3, 1, 3, 1], 2, 3),  # both neighbours equal and larger: the later one
        ([1, 4, 1, 3, 1], 2, 1),  # both larger: the larger one
    ],
)
def test_t1_climb_moves_to_the_local_maximum(column, start, expected) -> None:
    assert generator.climb_to_local_max(np.array(column, float), start) == expected


def test_t8_climb_between_upper_and_lower_reflection() -> None:
    """Risk record (section 2-1): a flank start climbs toward the rising side,
    even when that is the larger lower (pipe-interior) reflection."""
    upper, lower = 10, 20
    k = np.arange(31)
    column = 5 * np.exp(-(((k - upper) / 2.5) ** 2)) + 8 * np.exp(
        -(((k - lower) / 2.5) ** 2)
    )
    valley = upper + int(np.argmin(column[upper:lower]))
    assert generator.climb_to_local_max(column, valley + 1) == lower
    assert generator.climb_to_local_max(column, valley - 1) == upper


# ----------------------------------------------------------------- T2, T3


def test_t2_current_keeps_the_flank_start_and_a1_fixes_it() -> None:
    radargram, t_apex = t2_line()
    line, kept, _ = run(radargram)
    near = [s for s in kept if abs(s["trace"] - APEX_TRACE) <= WINDOW_TRACES]
    # The fixture makes the defect: a kept start on the rising flank, before the
    # apex window, whose start is not a time local maximum of E.
    assert near and not any(in_window(s, APEX_TRACE, t_apex) for s in near)
    assert all(s["t_ns"] < t_apex - 0.5 * PERIOD_NS for s in near)
    assert not any(peak_of(line, s)["start_is_local_max"] for s in near)

    line_a1, kept_a1, _ = run(radargram, **A1)
    fixed = [s for s in kept_a1 if in_window(s, APEX_TRACE, t_apex)]
    assert fixed
    assert all(peak_of(line_a1, s)["end_is_local_max"] for s in fixed)


def _window_fates(line, apex: int, t_apex: float) -> list[str]:
    return [
        fate["fate"]
        for item, fate in zip(
            line["_accepted_all"], line["_accepted_fate"], strict=True
        )
        if in_window(item, apex, t_apex)
    ]


def test_t3_current_caps_a_primary_and_a1b2_keeps_both() -> None:
    radargram, apex_times = t3_line()
    line, kept, _ = run(radargram)
    capped = [
        apex
        for apex, t_apex in apex_times.items()
        if not any(in_window(s, apex, t_apex) for s in kept)
        and _window_fates(line, apex, t_apex)
        and set(_window_fates(line, apex, t_apex)) == {generator.FATE_CAP6}
    ]
    assert capped, "fixture must make a primary lose to the per-line cap"

    _, kept_b2, _ = run(radargram, **A1B2)
    assert len(kept_b2) <= generator.MAX_APEX_CANDIDATES_PER_LINE
    for apex, t_apex in apex_times.items():
        assert any(in_window(s, apex, t_apex) for s in kept_b2)


def test_a1b1_has_no_cap_and_keeps_the_a1_ranking() -> None:
    radargram, _ = t3_line()
    _, kept_a1, _ = run(radargram, **A1)
    _, kept_b1, _ = run(radargram, start_rule="envelope_max", per_line_cap=None)
    assert len(kept_b1) > generator.MAX_APEX_CANDIDATES_PER_LINE
    assert kept_b1[: len(kept_a1)] == kept_a1


# ----------------------------------------------------------------- T4, T5


def _deep_envelope(noise_sigma: float = 0.0) -> np.ndarray:
    spec = BScanSpec(
        f_c_hz=F_C_HZ, dt_ns=DT_NS, dx_m=0.02, n_traces=161, n_samples=800,
        snr_db=None, d_top_m=1.5, eps_r=9.0,
    )  # fmt: skip
    data = make_bscan(spec).astype(np.float64)
    if noise_sigma:
        data += np.random.default_rng(20301112).normal(0.0, noise_sigma, data.shape)
    return np.abs(hilbert(data, axis=0))


@pytest.mark.parametrize("noise_sigma", [0.0, 0.05])
def test_t4_tau2_vertex_finds_the_apex_from_a_side_start(noise_sigma) -> None:
    envelope = _deep_envelope(noise_sigma)
    start_trace = APEX_TRACE + 5
    column = envelope[:, start_trace]
    k_start, _ = generator.parabolic_peak(column, int(np.argmax(column[300:])) + 300)
    points = generator.crest_walk(envelope, start_trace, k_start, DT_NS, 0.02)
    vertex = generator.tau2_vertex_trace(points, DT_NS, 0.02, T_SRC_NS)
    assert vertex is not None and abs(vertex - APEX_TRACE) <= 1
    # The walk is the one crest_apex takes: its earliest point is the output.
    earliest = min(points, key=lambda point: point[1])
    end = generator.crest_apex(envelope, start_trace, k_start, DT_NS, 0.02)
    assert (earliest[0], earliest[1]) == (end[0], end[1])


def test_t4_diagnostics_do_not_change_the_output() -> None:
    radargram, _ = t2_line()
    line, kept, _ = run(radargram)
    for peak in line["_peaks"]:
        assert {"sample_snapped", "start_is_local_max", "end_is_local_max",
                "tau2_vertex_trace"} <= set(peak)  # fmt: skip
    assert kept == run(radargram)[1]


def _parabola_points(traces, vertex_trace: int, curvature: float):
    dx = 0.02
    return [
        (int(j), (20.0 + curvature * ((j - vertex_trace) * dx) ** 2) / DT_NS)
        for j in traces
    ]


def test_t5_tau2_vertex_fallbacks_are_none() -> None:
    few = _parabola_points(range(78, 82), 80, 5.0)
    assert generator.tau2_vertex_trace(few, DT_NS, 0.02, 0.0) is None
    concave = _parabola_points(range(70, 91), 80, -5.0)
    assert generator.tau2_vertex_trace(concave, DT_NS, 0.02, 0.0) is None
    # Earliest point at the end of a one-sided crest: the vertex is 0.6 m away.
    one_sided = _parabola_points(range(0, 11), 40, 5.0)
    assert generator.tau2_vertex_trace(one_sided, DT_NS, 0.02, 0.0) is None
    non_positive = _parabola_points(range(70, 91), 80, 5.0)
    assert generator.tau2_vertex_trace(non_positive, DT_NS, 0.02, 30.0) is None
    good = _parabola_points(range(70, 91), 80, 5.0)
    assert generator.tau2_vertex_trace(good, DT_NS, 0.02, 0.0) == 80


# ----------------------------------------------------------------- T6, T7


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args], check=True, capture_output=True,
            text=True, encoding="utf-8",
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


@pytest.fixture(scope="module")
def generator_v1(tmp_path_factory) -> ModuleType:
    """The wrapper as sealed at 752e783 (blob 7cc2bf2b), loaded from git."""
    blob = _git("rev-parse", f"{V1_COMMIT}:{GENERATOR_PATH}")
    source = _git("show", f"{V1_COMMIT}:{GENERATOR_PATH}")
    if blob is None or source is None:
        pytest.skip("git history with 752e783 is not available")
    assert blob.startswith(V1_BLOB_PREFIX)
    path = tmp_path_factory.mktemp("generator_v1") / "fourtu_apex_candidate_v1.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("k7_generator_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NEW_PEAK_KEYS = {"sample_snapped", "start_is_local_max", "end_is_local_max",
                 "tau2_vertex_trace"}  # fmt: skip


@pytest.mark.parametrize("make_line", [t2_line, t3_line])
def test_t6_defaults_equal_the_sealed_generator(generator_v1, make_line) -> None:
    radargram, _ = make_line()
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    old_line, old_kept, old_envelope = generator_v1.apex_candidates_for_radargram(
        radargram, inputs, act="UNIT", path="line"
    )
    line, kept, envelope = generator.apex_candidates_for_radargram(
        radargram, inputs, act="UNIT", path="line"
    )
    assert kept == old_kept
    np.testing.assert_array_equal(envelope, old_envelope)
    for key in ("_accepted_all", "_accepted_fate"):
        assert line[key] == old_line[key]
    assert len(line["_peaks"]) == len(old_line["_peaks"])
    for new, old in zip(line["_peaks"], old_line["_peaks"], strict=True):
        assert set(new) - set(old) == NEW_PEAK_KEYS
        assert {key: new[key] for key in old} == old
    public = {k: v for k, v in line.items() if not k.startswith("_")}
    assert public == {k: v for k, v in old_line.items() if not k.startswith("_")}


@pytest.mark.parametrize(
    ("start_rule", "rank_rule", "per_line_cap"),
    [
        ("current", "snr", 6),
        ("current", "wings_snr", None),
        ("current", "snr", None),
        ("envelope_max", "snr", None),
        ("current", "wings_snr", 5),
        ("snap", "wings_snr", 6),
        ("envelope_max", "wings", 6),
    ],
)
def test_t7_other_combinations_raise(start_rule, rank_rule, per_line_cap) -> None:
    radargram, _ = t2_line()
    with pytest.raises(ValueError, match="not one of"):
        run(radargram, start_rule=start_rule, rank_rule=rank_rule,
            per_line_cap=per_line_cap)  # fmt: skip


def test_t7_the_four_allowed_combinations() -> None:
    assert generator.GENERATOR_VARIANTS == {
        "cur": ("current", "wings_snr", 6),
        "a1": ("envelope_max", "wings_snr", 6),
        "a1b2": ("envelope_max", "snr", 6),
        "a1b1": ("envelope_max", "wings_snr", None),
    }
    for name, combination in generator.GENERATOR_VARIANTS.items():
        assert generator.variant_of(*combination) == name


def _rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


@pytest.mark.data
def test_t6_fourtu_rows_byte_identical_on_two_activities(tmp_path, monkeypatch) -> None:
    """G0 on two activities (the full 759-line regeneration is in the harness)."""
    if not generator.BASE.exists():
        pytest.skip("4TU data not available")
    monkeypatch.setattr(generator, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["gen", "--activities", "01.1,04.1"])
    generator.main()
    committed_dir = PROJECT_ROOT / "docs" / "processing"
    for name in FOURTU_FILES:
        committed = _rows(committed_dir / name)
        regenerated = _rows(tmp_path / name)
        assert regenerated[0] == committed[0]
        subset = [row for row in committed[1:] if row[0] in {"01.1", "04.1"}]
        assert regenerated[1:] == subset
