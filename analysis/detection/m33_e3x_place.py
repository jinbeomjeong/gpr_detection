"""M3-3 pre-close item 3 (E3x): placement and field transforms (pure functions).

Spec: ``docs/processing/m33_e3x_plan.md`` v0.4 (v0.3 was seal 1, 72b0140), §2-3
(hyperbola positions, ``r_edge`` of the on/off difference field), §2-4 (band
rows by a pure vertical shift with one re-measured correction, one end taper
per position), §2-7 (texture fields) and appendix A-2.

Nothing here reads a file or runs a detector; the runner ``m33_e3x.py`` and
the tests share these definitions. Two pieces are copies of analysis scripts
outside the sealed E3x code (appendix A-1), asserted equal by the tests:

* :func:`untapered_field` and the off arm of :func:`inject_untapered` follow
  ``analysis/processing/m33_y0_diag_infer.inject_without_taper`` (the Y0 E3
  arm), extended to several positions per realization and to the int16 rule
  (plan §2-3: a rejected off field drops that pair);
* :func:`band_contrast` is ``analysis/processing/m33_r3_band_contrast.metrics``.

:func:`inject_bands` is ``inject.inject_band_realization`` for bands that are
already built (shifted, textured): it runs the same steps through the same
``inject`` helpers, and the tests assert it gives the same data and truths
as the library function on plain bands.

Seeded orders (plan §5-5 fixes ``PLAN_SEED_E3X`` and ``TEXTURE_SEED``; the
draw mechanics below belong to this implementation and are sealed with its
blob at seal 2):

* hyperbola model order of a background: one permutation of the sorted model
  ids from ``default_rng([PLAN_SEED_E3X, crc32(background)])``;
* hyperbola candidates of (background, model): one permutation of the
  candidates sorted by (line id, apex trace), from
  ``default_rng([PLAN_SEED_E3X, crc32(background), crc32(model)])``;
* band model order of a line: ``default_rng([PLAN_SEED_E3X, crc32(line)])``;
* band centre order of (line, model):
  ``default_rng([PLAN_SEED_E3X, crc32(line), crc32(model)])``;
* texture noise of a position: ``default_rng([TEXTURE_SEED, crc32(position)])``.
"""

import math
import zlib
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
from scipy.signal import fftconvolve, hilbert

from gpr_detection.eval import inject
from gpr_detection.io.radargram import GridBox

PLAN_SEED_E3X = 20261122
TEXTURE_SEED = 20261123

SHIFT_LIMIT_P = 12
"""Largest band shift in line periods (design constant, user decision v3-4)."""
SHIFT_MATCH_SAMPLES = 1.0
"""After the correction the truth top lies within one sample of the target (v0.4)."""
TARGET_ROWS_PX = (28.0, 47.0, 68.0, 90.0)
"""Target truth-top L1 rows (bins < 37 / 37-57 / 57-79 / >= 79 px, plan §2-2)."""
SHALLOW_ROW_PX = 28.0
IN_RANGE_ROWS_PX = (47.0, 68.0)
DEEP_ROW_PX = 90.0
REQUIRED_LONG_ROWS_PX = (28.0, 47.0, 68.0)
"""Rows every band position must reach with the long arm (HB1)."""
LONG_ARM = "long"
SHORT_ARM = "short"
ARM_PERIODS = {LONG_ARM: 13.0, SHORT_ARM: 6.0}
"""Band lengths ``l`` in background-image periods (plan §2-2)."""
SHORT_ARM_ROWS_PX = IN_RANGE_ROWS_PX
MAX_L_OVER_W_EFF = 0.45
MAX_BAND_POSITIONS_PER_LINE = 2

MAX_HYPERBOLA_POSITIONS_PER_LINE = 4
MAX_HYPERBOLA_POSITIONS_PER_GROUP = 2
MAX_HYPERBOLA_POSITIONS_PER_MODEL_LINE = 2
"""A second position of one model on one line only after the first pass."""
HYPERBOLA_GROUPS = 2

TEXTURE_CORRELATION_PERIODS = 3.0
"""``l_c``: standard deviation of the Gaussian kernel in periods (plan §2-7)."""
AMPLITUDE_TEXTURE_DB = 2.5
JITTER_PERIODS = (0.1, 0.2)
KERNEL_HALF_WIDTH_SIGMAS = 4.0
"""The Gaussian kernel is cut at +-4 standard deviations (weight < 3.4e-4)."""

JUDGED_SNR_DB = (12.0, 18.0)
MANIPULATED_MIN_DB = 0.0
"""A unit is manipulated when ``r_edge + s >= 0`` dB (plan §2-3)."""
NEGATIVE_CONTROL_SNR_DB = 18.0
NEGATIVE_CONTROL_BELOW_DB = -6.0
"""A position is a negative control when ``r_edge + 18 < -6`` dB."""


class EdgeStratum(StrEnum):
    """Position stratum of the aperture-end manipulation (plan §2-3)."""

    MANIPULATED = "manipulated"
    GRAY = "gray"
    NEGATIVE_CONTROL = "negative_control"


class ShiftReject(StrEnum):
    """Band shift failures that are not ``inject.RejectReason`` values."""

    SHIFT_LIMIT = "shift_limit"
    SHIFT_MISMATCH = "shift_mismatch"


class MainCellReject(StrEnum):
    """A band arm outside the C3-6 main cell (plan §2-4 eligibility)."""

    BOX_CLIPPED = "box_clipped"
    WING_PAST_RECORD_END = "wing_past_record_end"


Reject = inject.RejectReason | ShiftReject | MainCellReject


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


# ---------------------------------------------------------- edge strata


def edge_snr_set(r_edge: float) -> tuple[float, ...]:
    """``S_p^edge``: judged SNR steps with ``r_edge + s >= 0`` dB (boundary in)."""
    return tuple(s for s in JUDGED_SNR_DB if r_edge + s >= MANIPULATED_MIN_DB)


def edge_stratum(r_edge: float) -> EdgeStratum:
    """Stratum of a position from its own ``r_edge(p)`` (user decision v3-3).

    ``r_edge + 18 = -6`` dB exactly is grey (the control needs ``< -6``).
    """
    if edge_snr_set(r_edge):
        return EdgeStratum.MANIPULATED
    if r_edge + NEGATIVE_CONTROL_SNR_DB < NEGATIVE_CONTROL_BELOW_DB:
        return EdgeStratum.NEGATIVE_CONTROL
    return EdgeStratum.GRAY


def envelope(section: np.ndarray) -> np.ndarray:
    """Hilbert envelope along time, zero padded to ``2 N`` (as ``inject``)."""
    n_rows = section.shape[0]
    return np.abs(hilbert(section, N=2 * n_rows, axis=0)[:n_rows])


def untapered_field(patch: inject.ResponsePatch) -> np.ndarray:
    """``patch.field`` with the aperture taper divided out (window-end taper kept)."""
    weights = inject.edge_taper_weights(patch.response.n_traces, patch.n_aperture_taper)
    return patch.field / weights[None, :]


def apex_envelope_peak(
    values: np.ndarray, apex_column: int, apex_sample: float, period: float
) -> float:
    """Envelope maximum of the apex column within apex +- 1 P (the ``A`` window)."""
    row0 = max(0, math.floor(apex_sample - period))
    row1 = min(values.shape[0] - 1, math.ceil(apex_sample + period))
    return float(values[row0 : row1 + 1, apex_column].max())


def r_edge_db(patch: inject.ResponsePatch, context: inject.LineContext) -> float:
    """Manipulation strength of the aperture end (plan v0.4 §2-3, decision v4-2).

    ``20 log10(max over every row and column of e~ / e_apex)``: ``e~`` is
    the envelope, after the line's steps (1)-(3), of the field by which the
    off and on arms differ, ``untapered_field(patch) - patch.field`` =
    ``f_unt (1 - w)`` (zero outside the aperture-taper columns). ``e_apex``
    is the envelope maximum of the untapered field in the apex column within
    apex +- 1 ``P_line`` (the ``signal_amplitude`` window; ``w`` is 1 there).

    Raises:
        ValueError: If the patch has no aperture-taper column.
    """
    if patch.n_aperture_taper <= 0:
        raise ValueError(f"{patch.response.response_id}: no aperture-taper column")
    background = context.background
    untapered = untapered_field(patch)
    difference = envelope(inject.linear_steps(untapered - patch.field, background))
    apex = apex_envelope_peak(
        envelope(inject.linear_steps(untapered, background)),
        round(patch.response.apex_trace),
        patch.apex_sample,
        context.period_samples,
    )
    return 20.0 * math.log10(float(difference.max()) / apex)


# ------------------------------------------------------ hyperbola planning


@dataclass(frozen=True)
class HyperbolaCandidate:
    """One allowed (model, apex) of a background line (zones (a)-(d), main cell)."""

    line_id: str
    background: str
    model: str
    apex_trace: float
    half_width_20db: float
    period_traces: float


@dataclass(frozen=True)
class PlannedHyperbola:
    """A chosen hyperbola position and its realization group (0 or 1)."""

    line_id: str
    background: str
    model: str
    apex_trace: float
    half_width_20db: float
    group: int
    second_on_line: bool


@dataclass(frozen=True)
class HyperbolaPlan:
    """Chosen positions, in choice order, and the targets left unmet."""

    positions: tuple[PlannedHyperbola, ...]
    shortfall: Mapping[tuple[str, str], int] = field(default_factory=dict)


@dataclass
class _LineSlots:
    groups: list[list[tuple[float, float]]] = field(
        default_factory=lambda: [[] for _ in range(HYPERBOLA_GROUPS)]
    )
    per_model: dict[str, int] = field(default_factory=dict)

    @property
    def placed(self) -> list[tuple[float, float]]:
        return [apex for group in self.groups for apex in group]


def _fitting_group(slots: _LineSlots, candidate: HyperbolaCandidate) -> int | None:
    """First group with room where the realization spacing (e') holds."""
    for index, members in enumerate(slots.groups):
        if len(members) >= MAX_HYPERBOLA_POSITIONS_PER_GROUP:
            continue
        if inject.realization_spacing_ok(
            candidate.apex_trace,
            candidate.half_width_20db,
            members,
            candidate.period_traces,
        ):
            return index
    return None


def _eligible_group(slots: _LineSlots, candidate: HyperbolaCandidate) -> int | None:
    if len(slots.placed) >= MAX_HYPERBOLA_POSITIONS_PER_LINE:
        return None
    if not inject.line_spacing_ok(
        candidate.apex_trace, candidate.half_width_20db, slots.placed
    ):
        return None
    return _fitting_group(slots, candidate)


def candidate_order(
    candidates: Sequence[HyperbolaCandidate], background: str, model: str, seed: int
) -> list[HyperbolaCandidate]:
    """Seeded visiting order of one model's candidates in one background."""
    ordered = sorted(candidates, key=lambda c: (c.line_id, c.apex_trace))
    rng = np.random.default_rng([seed, _crc(background), _crc(model)])
    return [ordered[int(i)] for i in rng.permutation(len(ordered))]


def model_order(models: Sequence[str], background: str, seed: int) -> list[str]:
    """Seeded round-robin order of the models of one background."""
    ordered = sorted(models)
    rng = np.random.default_rng([seed, _crc(background)])
    return [ordered[int(i)] for i in rng.permutation(len(ordered))]


def _pick(
    ordered: Sequence[HyperbolaCandidate],
    lines: Mapping[str, _LineSlots],
    model: str,
) -> tuple[HyperbolaCandidate, int, bool] | None:
    """First eligible candidate: lines without the model, then a second one."""
    for allowed_count, second in ((0, False), (1, True)):
        for candidate in ordered:
            slots = lines[candidate.line_id]
            if slots.per_model.get(model, 0) != allowed_count:
                continue
            group = _eligible_group(slots, candidate)
            if group is not None:
                return candidate, group, second
    return None


def plan_hyperbola_positions(
    candidates: Sequence[HyperbolaCandidate],
    targets: Mapping[tuple[str, str], int],
    seed: int = PLAN_SEED_E3X,
) -> HyperbolaPlan:
    """Round-robin hyperbola positions (plan §2-3, steps 1-4).

    Per background, the models take turns in a seeded order, one position
    per turn. A model's turn takes its first eligible candidate (seeded
    order) on a line where it has no position yet; only when there is none
    does it take a second position on a line it already has. Eligible means
    the line spacing ``W_-20 / 2`` to every chosen position of the line and
    a realization group (at most two per line, two positions per group) whose
    spacing ``W_-20 + 1 P`` holds. A model leaves the turn order when it
    reaches its target or has no eligible candidate (the shortfall is kept).

    Args:
        candidates: Allowed (model, apex) pairs of every background line.
        targets: ``(background, model) -> positions`` (A0: 4 + 4, A1: 8 on 4TU).
        seed: ``PLAN_SEED_E3X``.
    """
    lines = {c.line_id: _LineSlots() for c in candidates}
    chosen: list[PlannedHyperbola] = []
    shortfall: dict[tuple[str, str], int] = {}
    for background in sorted({key[0] for key in targets}):
        wanted = {m: n for (bg, m), n in targets.items() if bg == background and n > 0}
        by_model = {
            model: candidate_order(
                [
                    c
                    for c in candidates
                    if c.background == background and c.model == model
                ],
                background,
                model,
                seed,
            )
            for model in wanted
        }
        placed = dict.fromkeys(wanted, 0)
        turns = deque(model_order(list(wanted), background, seed))
        while turns:
            model = turns.popleft()
            picked = _pick(by_model[model], lines, model)
            if picked is None:
                shortfall[(background, model)] = wanted[model] - placed[model]
                continue
            candidate, group, second = picked
            slots = lines[candidate.line_id]
            slots.groups[group].append(
                (candidate.apex_trace, candidate.half_width_20db)
            )
            slots.per_model[model] = slots.per_model.get(model, 0) + 1
            placed[model] += 1
            chosen.append(
                PlannedHyperbola(
                    candidate.line_id,
                    background,
                    model,
                    candidate.apex_trace,
                    candidate.half_width_20db,
                    group,
                    second,
                )
            )
            if placed[model] < wanted[model]:
                turns.append(model)
    return HyperbolaPlan(tuple(chosen), shortfall)


# ------------------------------------------------------------ band shifts


def shifted_response(
    response: inject.PipeResponse, delta_t_ns: float
) -> inject.PipeResponse:
    """The response moved later by ``delta_t_ns`` (positive = deeper row).

    Only the time reference moves (plan §2-4, A-2): ``top_depth_m``, the
    2.5-D weight and the waveform are those of the original model.
    """
    return replace(response, reference_time_ns=response.reference_time_ns - delta_t_ns)


def position_taper_traces(
    response: inject.PipeResponse, context: inject.LineContext
) -> int:
    """End taper ``T`` of a band position, from the original response (§2-4)."""
    return inject.band_taper_traces(response, context)


def band_length_traces(context: inject.LineContext, length_periods: float) -> int:
    """``ceil(l P_ell tau_x)``: a band length in background-image periods."""
    return math.ceil(length_periods * context.period_traces)


def target_sample(context: inject.LineContext, row_px: float) -> float:
    """L0 sample of an L1 row of the background image (``ImageGridMap``)."""
    grid = context.grid_map
    return grid.sample_offset + row_px * grid.samples_per_px


def row_px_of(context: inject.LineContext, sample: float) -> float:
    """L1 row of an L0 sample edge on the background image."""
    grid = context.grid_map
    return (sample - grid.sample_offset) / grid.samples_per_px


@dataclass(frozen=True)
class RowShift:
    """One model moved so its truth top lands on a target row (plan v0.4 §2-4).

    ``n1 = round(k* - k1_0)``; the band is built at ``n1`` and its top
    ``k1'(n1)`` re-measured; ``n2 = n1 + round(k* - k1'(n1))`` is the final
    shift (one correction, never repeated).

    Attributes:
        target_row_px: ``y*``.
        shift_samples: ``n2`` (every shift quantity below uses it).
        shift_samples_first: ``n1``.
        shift_step_samples: ``k1'(n1) - (k1_0 + n1)``: the re-measured step.
        shift_gap_samples: ``k1'(n2) - k*``.
        delta_t_ns: ``dt_line * n2``.
        shift_periods: ``|n2| / P_line``.
        native_row0: ``k1_0``, the truth top sample before the shift.
        native_y1_px: ``y1_0``, the same on the background image.
        shifted_row0: ``k1'(n2)``.
        response: The response moved by ``n2``.
        patch: Its S4 patch on the line (k 1.3).
    """

    target_row_px: float
    shift_samples: int
    shift_samples_first: int
    shift_step_samples: int
    shift_gap_samples: float
    delta_t_ns: float
    shift_periods: float
    native_row0: int
    native_y1_px: float
    shifted_row0: int
    response: inject.PipeResponse
    patch: inject.ResponsePatch

    @property
    def shift_correction_samples(self) -> int:
        """``n2 - n1``."""
        return self.shift_samples - self.shift_samples_first

    @property
    def row_offset(self) -> int:
        """``k1'(n2) - n2``: equal on every arm of a position (assertion (ii))."""
        return self.shifted_row0 - self.shift_samples


def shift_samples_for(
    context: inject.LineContext, target_row_px: float, native_row0: int
) -> int:
    """Whole-sample shift that puts ``row0`` on the target row (Python round)."""
    return round(target_sample(context, target_row_px) - native_row0)


def within_shift_limit(shift_samples: int, period_samples: float) -> bool:
    """``|shift| <= 12 P_line`` (product form; exactly 12 P is allowed)."""
    return abs(shift_samples) <= SHIFT_LIMIT_P * period_samples


@dataclass(frozen=True)
class _ShiftedTop:
    response: inject.PipeResponse
    patch: inject.ResponsePatch
    row0: int


def _shifted_top(
    context: inject.LineContext,
    response: inject.PipeResponse,
    k: float,
    shift: int,
    length_traces: int,
    taper_traces: int,
) -> _ShiftedTop | inject.RejectReason:
    """Response moved by ``shift`` samples, its patch and its band top ``k1'``."""
    moved = shifted_response(response, shift * context.dt_ns)
    patch = inject.build_response_patch(moved, context, k)
    if isinstance(patch, inject.RejectReason):
        return patch
    band = inject.build_band_patch(patch, context, length_traces, taper_traces)
    if isinstance(band, inject.RejectReason):
        return band
    return _ShiftedTop(moved, patch, band.row0)


def row_shift_for(
    context: inject.LineContext,
    response: inject.PipeResponse,
    k: float,
    target_row_px: float,
    native_row0: int,
    taper_traces: int,
    length_traces: int,
) -> RowShift | Reject:
    """Shift a model to a target row with one re-measured correction (v0.4).

    A failure of the ``n1`` build returns its reason without a correction;
    ``|n2| > 12 P`` is ``SHIFT_LIMIT`` (``n1`` is not limited). With a zero
    correction the ``n1`` build is the result (one build), else the model
    is built once more at ``n2``.
    """
    k_target = target_sample(context, target_row_px)
    first_shift = shift_samples_for(context, target_row_px, native_row0)
    first = _shifted_top(context, response, k, first_shift, length_traces, taper_traces)
    if isinstance(first, inject.RejectReason):
        return first
    final_shift = first_shift + round(k_target - first.row0)
    if not within_shift_limit(final_shift, context.period_samples):
        return ShiftReject.SHIFT_LIMIT
    final = first
    if final_shift != first_shift:
        rebuilt = _shifted_top(
            context, response, k, final_shift, length_traces, taper_traces
        )
        if isinstance(rebuilt, inject.RejectReason):
            return rebuilt
        final = rebuilt
    return RowShift(
        target_row_px=target_row_px,
        shift_samples=final_shift,
        shift_samples_first=first_shift,
        shift_step_samples=first.row0 - (native_row0 + first_shift),
        shift_gap_samples=final.row0 - k_target,
        delta_t_ns=final_shift * context.dt_ns,
        shift_periods=abs(final_shift) / context.period_samples,
        native_row0=native_row0,
        native_y1_px=row_px_of(context, native_row0),
        shifted_row0=final.row0,
        response=final.response,
        patch=final.patch,
    )


def shifted_band(
    context: inject.LineContext,
    row: RowShift,
    length_traces: int,
    taper_traces: int,
) -> inject.BandPatch | Reject:
    """The band of a shifted model; ``SHIFT_MISMATCH`` when the top misses ``y*``.

    Check (i) of plan v0.4 §2-4: after the correction the top lies within
    ``SHIFT_MATCH_SAMPLES`` (one sample, boundary included) of ``k*``.
    """
    band = inject.build_band_patch(row.patch, context, length_traces, taper_traces)
    if isinstance(band, inject.RejectReason):
        return band
    gap = abs(band.row0 - target_sample(context, row.target_row_px))
    if gap > SHIFT_MATCH_SAMPLES:
        return ShiftReject.SHIFT_MISMATCH
    return band


def native_row0(
    context: inject.LineContext,
    patch: inject.ResponsePatch,
    length_traces: int,
    taper_traces: int,
) -> int | inject.RejectReason:
    """Truth top row of the unshifted band (independent of the length)."""
    band = inject.build_band_patch(patch, context, length_traces, taper_traces)
    if isinstance(band, inject.RejectReason):
        return band
    return band.row0


def offsets_parallel(rows: Sequence[RowShift]) -> bool:
    """Check (ii): ``k1'(n2) - n2`` is one integer over every arm of a position."""
    return len({row.row_offset for row in rows}) <= 1


def band_shift_for(
    context: inject.LineContext,
    response: inject.PipeResponse,
    k: float,
    target_y1_px: float,
    taper_traces: int,
    length_traces: int,
) -> tuple[RowShift, inject.BandPatch] | Reject:
    """Appendix A-2 ``band_shift_for`` (v0.4): ``k1_0``, ``n1``, re-measure, ``n2``.

    ``length_traces`` is an argument here (the band needs a length); the
    truth top does not depend on it.
    """
    native = inject.build_response_patch(response, context, k)
    if isinstance(native, inject.RejectReason):
        return native
    row0 = native_row0(context, native, length_traces, taper_traces)
    if isinstance(row0, inject.RejectReason):
        return row0
    row = row_shift_for(
        context, response, k, target_y1_px, row0, taper_traces, length_traces
    )
    if not isinstance(row, RowShift):
        return row
    band = shifted_band(context, row, length_traces, taper_traces)
    if not isinstance(band, inject.BandPatch):
        return band
    return row, band


def d_prime_m(context: inject.LineContext, band: inject.BandPatch) -> float:
    """``v_line (t'_apex - t_z) / 2``: recorded only, used nowhere (plan §2-4)."""
    t_zero_ns = context.zero.t_zero_ns
    assert t_zero_ns is not None  # patches are rejected on a failed zero
    velocity = inject.line_velocity_m_per_ns(context.background)
    return velocity * (band.apex_sample * context.dt_ns - t_zero_ns) / 2.0


def main_cell_failure(
    context: inject.LineContext, band: inject.BandPatch
) -> MainCellReject | None:
    """C3-6 main cell of a band: not cut at the record end, wing before end - 1 P."""
    if band.row1 > context.background.n_samples - 1:
        return MainCellReject.BOX_CLIPPED
    if inject.band_wing_past_record_end(band, context):
        return MainCellReject.WING_PAST_RECORD_END
    return None


def arm_failure(
    context: inject.LineContext, band: inject.BandPatch, center_trace: float
) -> Reject | None:
    """Zones (a)-(d) of the band at a centre, then the main cell."""
    failure = inject.band_placement_failure(context, band, center_trace)
    if failure is not None:
        return failure
    return main_cell_failure(context, band)


def short_arm_center(
    long_center: float, long_columns: int, short_columns: int
) -> float:
    """Centre of the short arm on the trace grid nearest the long-arm centre.

    The two arms have ``L + T`` columns each (``T`` even, so the parity is that
    of ``L``); when the column counts differ in parity the same centre cannot
    lie on the trace grid for both, so the short arm takes the long-arm
    centre minus 0.5 trace (plan v0.4 §2-4, user decision v4-3: about
    0.08 P on 4TU; the short arm stays inside the long arm, and its zones and
    truth box use its own centre). The runner records the offset
    (``short_center_offset_traces``).
    """
    if (long_columns - short_columns) % 2 == 0:
        return long_center
    return long_center - 0.5


# ------------------------------------------------------------ band planning


@dataclass(frozen=True)
class BandArm:
    """One row x length arm of a band position (fixed model, centre and T)."""

    row_px: float
    arm: str
    center_trace: float
    length_traces: int
    shift: RowShift
    band: inject.BandPatch


@dataclass(frozen=True)
class PlannedBandPosition:
    """A band position: model, centre, taper and every arm it reaches."""

    line_id: str
    index: int
    model: str
    center_trace: float
    taper_traces: int
    arms: tuple[BandArm, ...]

    @property
    def has_deep_row(self) -> bool:
        """The 90 px long arm passed (else the position leaves HB1b only)."""
        return any(a.arm == LONG_ARM and a.row_px == DEEP_ROW_PX for a in self.arms)

    def arm_of(self, row_px: float, arm: str) -> BandArm | None:
        """The arm at a row and length, ``None`` when it is not planned."""
        for item in self.arms:
            if item.row_px == row_px and item.arm == arm:
                return item
        return None


@dataclass(frozen=True)
class BandLinePlan:
    """Band positions of one line and why models or the line were left out."""

    line_id: str
    positions: tuple[PlannedBandPosition, ...]
    lengths: Mapping[str, int]
    excluded_reason: str | None = None
    model_reasons: Mapping[str, str] = field(default_factory=dict)
    shift_mismatches: tuple[dict, ...] = ()
    arm_failures: Mapping[str, int] = field(default_factory=dict)
    """First failing required arm per tried (model, centre): report only."""

    @property
    def n_shift_mismatch(self) -> int:
        """Shifted tops more than one sample from their target (defects)."""
        return len(self.shift_mismatches)


@dataclass(frozen=True)
class _ModelBands:
    model: str
    taper_traces: int
    long: Mapping[float, BandArm]
    short: Mapping[float, BandArm]


def _model_arms(
    context: inject.LineContext,
    response: inject.PipeResponse,
    lengths: Mapping[str, int],
) -> tuple[_ModelBands | None, str | None, list[dict]]:
    """Shifted bands of one model at every row, or why the model is out.

    The ``T`` rules (``L_long + 2 T <= N``, ``L_short > T``) are applied per
    (line, model) because ``T`` follows the model's top depth (plan v0.4
    §2-4 step 1, user decision v4-4). The 12 P limit is checked on the
    corrected shift ``n2`` of each row. Check (i) is applied to every arm
    and check (ii) (one ``k1'(n2) - n2`` over every arm) to the model; a
    failure of either excludes the model and is a defect (exit 1).

    Returns:
        ``(bands, reason, shift mismatches seen)``; ``bands`` is ``None``
        when one of the three required long rows or a short row fails. Each
        mismatch is a dict for the plan report.
    """
    patch = inject.build_response_patch(response, context, inject.K_MAIN)
    if isinstance(patch, inject.RejectReason):
        return None, patch.value, []
    taper = position_taper_traces(response, context)
    n_traces = context.background.n_traces
    if lengths[LONG_ARM] + 2 * taper > n_traces:
        return None, "long_arm_plus_two_tapers_over_line", []
    if lengths[SHORT_ARM] <= taper:
        return None, "short_arm_not_longer_than_taper", []
    row0 = native_row0(context, patch, lengths[LONG_ARM], taper)
    if isinstance(row0, inject.RejectReason):
        return None, row0.value, []
    long_arms: dict[float, BandArm] = {}
    short_arms: dict[float, BandArm] = {}
    mismatches: list[dict] = []
    for row_px in TARGET_ROWS_PX:
        row = row_shift_for(
            context, response, inject.K_MAIN, row_px, row0, taper, lengths[LONG_ARM]
        )
        if not isinstance(row, RowShift):
            if row_px in REQUIRED_LONG_ROWS_PX:
                return None, f"row{row_px:g}:{row.value}", mismatches
            continue
        for arm, arms in ((LONG_ARM, long_arms), (SHORT_ARM, short_arms)):
            if arm == SHORT_ARM and row_px not in SHORT_ARM_ROWS_PX:
                continue
            band = shifted_band(context, row, lengths[arm], taper)
            if band is ShiftReject.SHIFT_MISMATCH:
                mismatches.append(_mismatch_record(context, row, arm, band=None))
            if not isinstance(band, inject.BandPatch):
                if arm == LONG_ARM and row_px == DEEP_ROW_PX:
                    continue
                return None, f"{arm}{row_px:g}:{band.value}", mismatches
            arms[row_px] = BandArm(row_px, arm, 0.0, lengths[arm], row, band)
    every_arm = [*long_arms.values(), *short_arms.values()]
    if not offsets_parallel([arm.shift for arm in every_arm]):
        for arm in every_arm:
            mismatches.append(
                _mismatch_record(context, arm.shift, arm.arm, arm.band, parallel=False)
            )
        return None, "arm_offsets_not_parallel", mismatches
    bands = _ModelBands(response.response_id, taper, long_arms, short_arms)
    return bands, None, mismatches


def _mismatch_record(
    context: inject.LineContext,
    row: RowShift,
    arm: str,
    band: inject.BandPatch | None,
    parallel: bool = True,
) -> dict:
    """A check (i) or (ii) failure for the plan report (v0.4 fields)."""
    return {
        "model": row.response.response_id,
        "row_px": row.target_row_px,
        "arm": arm,
        "native_row0": row.native_row0,
        "shift_samples_first": row.shift_samples_first,
        "shift_samples": row.shift_samples,
        "shift_correction_samples": row.shift_correction_samples,
        "shift_step_samples": row.shift_step_samples,
        "shift_gap_samples": row.shift_gap_samples,
        "shifted_row0": row.shifted_row0 if band is None else band.row0,
        "target_sample": target_sample(context, row.target_row_px),
        "row_offset": row.row_offset,
        "arm_offsets_parallel": parallel,
    }


def _arms_at(
    context: inject.LineContext,
    bands: _ModelBands,
    long_center: float,
    failures: Counter[str],
) -> tuple[BandArm, ...] | None:
    """Every arm placed at one centre; ``None`` if a required arm fails.

    The first failing required arm's reason is counted in ``failures``
    (plan report diagnostic only).
    """
    placed: list[BandArm] = []
    long_columns = next(iter(bands.long.values())).band.n_columns
    short_columns = next(iter(bands.short.values())).band.n_columns
    short_center = short_arm_center(long_center, long_columns, short_columns)
    required = [(bands.long[r], long_center) for r in REQUIRED_LONG_ROWS_PX]
    required += [(bands.short[r], short_center) for r in SHORT_ARM_ROWS_PX]
    for arm, center in required:
        failure = arm_failure(context, arm.band, center)
        if failure is not None:
            failures[f"{arm.arm}{arm.row_px:g}:{failure.value}"] += 1
            return None
        placed.append(replace(arm, center_trace=center))
    deep = bands.long.get(DEEP_ROW_PX)
    if deep is not None and arm_failure(context, deep.band, long_center) is None:
        placed.append(replace(deep, center_trace=long_center))
    return tuple(placed)


def _center_order(
    context: inject.LineContext, bands: _ModelBands, line_id: str, seed: int
) -> list[float]:
    """Seeded order of the long-arm centres on the trace grid."""
    n_columns = next(iter(bands.long.values())).band.n_columns
    n_offsets = context.background.n_traces - n_columns + 1
    centres = [offset + (n_columns - 1) / 2.0 for offset in range(max(0, n_offsets))]
    rng = np.random.default_rng([seed, _crc(line_id), _crc(bands.model)])
    return [centres[int(i)] for i in rng.permutation(len(centres))]


def _spaced_from(
    center: float,
    length_traces: int,
    chosen: Sequence[PlannedBandPosition],
    w_eff: int,
) -> bool:
    """Band spacing (e) ``|c1 - c2| >= (L1 + L2) / 2 + W_eff`` to every position."""
    placed = [
        (position.center_trace, position.arm_of(SHALLOW_ROW_PX, LONG_ARM).length_traces)
        for position in chosen
    ]
    return inject.band_spacing_ok(center, length_traces, placed, w_eff)


def plan_band_positions_e3x(
    context: inject.LineContext,
    responses: Mapping[str, inject.PipeResponse],
    seed: int = PLAN_SEED_E3X,
) -> BandLinePlan:
    """Band positions of one line, plan v0.4 §2-4 selection steps 1-5.

    1. ``L_long = ceil(13 P tau)``, ``L_short = ceil(6 P tau)``; the line is
       out when ``L_long > 0.45 W_eff``. The two taper rules
       (``L_long + 2 T <= N``, ``L_short > T``) are per (line, model)
       (user decision v4-4).
    2. Models in seeded order whose three required rows need at most 12 P
       after the correction (``n2``).
    3. Centres in seeded order: the first where the long arm passes 28, 47
       and 68 px and the short arm 47 and 68 px (zones and main cell); 90 px
       joins when it passes.
    4. No such centre: the next model.
    5. A second position, same steps, at least ``(L1 + L2) / 2 + W_eff``
       from the first.
    """
    line_id = context.background.line_id
    lengths = {arm: band_length_traces(context, p) for arm, p in ARM_PERIODS.items()}
    w_eff = inject.effective_median_window(context)
    if lengths[LONG_ARM] > MAX_L_OVER_W_EFF * w_eff:
        return BandLinePlan(line_id, (), lengths, "long_arm_over_0.45_w_eff")
    rng = np.random.default_rng([seed, _crc(line_id)])
    names = sorted(responses)
    ordered = [names[int(i)] for i in rng.permutation(len(names))]
    reasons: dict[str, str] = {}
    usable: list[tuple[_ModelBands, list[float]]] = []
    mismatches: list[dict] = []
    for model in ordered:
        bands, reason, seen = _model_arms(context, responses[model], lengths)
        mismatches += seen
        if bands is None:
            reasons[model] = reason or "unknown"
            continue
        usable.append((bands, _center_order(context, bands, line_id, seed)))
    chosen: list[PlannedBandPosition] = []
    failures: Counter[str] = Counter()
    while len(chosen) < MAX_BAND_POSITIONS_PER_LINE:
        position = _next_position(context, usable, chosen, w_eff, failures)
        if position is None:
            break
        chosen.append(position)
    if not chosen and not reasons:
        reasons["(line)"] = "no_model"
    return BandLinePlan(
        line_id,
        tuple(chosen),
        lengths,
        None,
        reasons,
        tuple(mismatches),
        dict(failures),
    )


def _next_position(
    context: inject.LineContext,
    usable: Sequence[tuple[_ModelBands, list[float]]],
    chosen: Sequence[PlannedBandPosition],
    w_eff: int,
    failures: Counter[str],
) -> PlannedBandPosition | None:
    for bands, centres in usable:
        long_length = bands.long[SHALLOW_ROW_PX].length_traces
        for centre in centres:
            if not _spaced_from(centre, long_length, chosen, w_eff):
                continue
            arms = _arms_at(context, bands, centre, failures)
            if arms is None:
                continue
            return PlannedBandPosition(
                context.background.line_id,
                len(chosen),
                bands.model,
                centre,
                bands.taper_traces,
                arms,
            )
    return None


# ------------------------------------------------------------- textures


def texture_field(
    n_columns: int,
    flat_columns: tuple[int, int],
    period_traces: float,
    corr_periods: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``w``: Gaussian-smoothed noise, mean 0 and SD 1 on the flat columns (§2-7).

    ``z`` is one standard normal value per band column; the kernel standard
    deviation is ``l_c = corr_periods x period_traces`` traces (the process
    autocorrelation then has SD ``sqrt(2) l_c``). The convolution is an FFT
    filter (``fftconvolve``, same length, kernel cut at +-4 SD).

    Args:
        n_columns: Band columns ``L + T`` (tapers included).
        flat_columns: First and last flat column (``BandPatch.flat_columns``).
        period_traces: Line period in traces.
        corr_periods: ``l_c`` in periods (3).
        rng: Position generator (``[TEXTURE_SEED, crc32(position_id)]``).
    """
    sigma = corr_periods * period_traces
    half = math.ceil(KERNEL_HALF_WIDTH_SIGMAS * sigma)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    noise = rng.standard_normal(n_columns)
    smoothed = fftconvolve(noise, kernel / kernel.sum(), mode="same")
    first, last = flat_columns
    flat = smoothed[first : last + 1]
    return (smoothed - flat.mean()) / flat.std()


def amplitude_texture(
    w: np.ndarray, flat_columns: tuple[int, int], sigma_db: float
) -> np.ndarray:
    """``m(x) = 10^(g/20) / sqrt(mean_flat 10^(g/10))``, ``g = sigma_db w``."""
    g = sigma_db * w
    first, last = flat_columns
    power = 10.0 ** (g[first : last + 1] / 10.0)
    return 10.0 ** (g / 20.0) / math.sqrt(float(power.mean()))


def jitter_delays(
    w: np.ndarray, sigma_periods: float, period_samples: float
) -> np.ndarray:
    """``delta(x) = sigma_delta P_line w(x)`` in samples (positive = later)."""
    return sigma_periods * period_samples * w


def time_jitter(
    column: np.ndarray, w: np.ndarray, sigma_periods: float, period_samples: float
) -> np.ndarray:
    """The apex column delayed by ``delta(x)`` per band column (FFT phase shift).

    A circular phase rotation over the column length: the column is already
    low-passed to 0.8 Nyquist, so the band-limited shift is exact and its
    energy is kept.

    Returns:
        ``[n_rows, len(w)]``: one shifted copy of ``column`` per band column.
    """
    n_rows = column.shape[0]
    delays = jitter_delays(w, sigma_periods, period_samples)
    spectrum = np.fft.rfft(np.asarray(column, dtype=np.float64))
    frequencies = np.fft.rfftfreq(n_rows)
    phases = np.exp(-2j * np.pi * np.outer(frequencies, delays))
    return np.fft.irfft(spectrum[:, None] * phases, n=n_rows, axis=0)


def apex_column(band: inject.BandPatch) -> np.ndarray:
    """The repeated apex column of a band (weight 1 on the flat columns)."""
    return band.patch.field[:, round(band.patch.response.apex_trace)]


def amplitude_textured_band(
    band: inject.BandPatch, multipliers: np.ndarray
) -> inject.BandPatch:
    """The band with ``m(x)`` on its columns (truth rows and ``A`` unchanged)."""
    return replace(band, field=band.field * multipliers[None, :])


def jittered_band(
    band: inject.BandPatch, w: np.ndarray, sigma_periods: float, period_samples: float
) -> inject.BandPatch:
    """The band whose columns are the apex column delayed by ``delta(x)``.

    The end taper weights are applied after the shift; the truth rows stay
    those of the unshifted column (plan §2-7).
    """
    weights = inject.edge_taper_weights(band.n_columns, band.taper_traces)
    shifted = time_jitter(apex_column(band), w, sigma_periods, period_samples)
    return replace(band, field=shifted * weights[None, :])


@dataclass(frozen=True)
class TextureCheck:
    """Precheck 5 errors of one position's texture fields (relative)."""

    w_mean: float
    w_sd: float
    amplitude_mean_square: float
    jitter_sd: dict[float, float]
    jitter_energy: dict[float, float]

    def passes(self) -> bool:
        """Every error within its precheck 5 tolerance."""
        return (
            self.w_mean <= 1e-12
            and self.w_sd <= 1e-12
            and self.amplitude_mean_square <= 1e-12
            and all(value <= 1e-12 for value in self.jitter_sd.values())
            and all(value <= 1e-9 for value in self.jitter_energy.values())
        )


def texture_check(
    band: inject.BandPatch, w: np.ndarray, multipliers: np.ndarray, period: float
) -> TextureCheck:
    """Plan §5-2 precheck 5 on one position (also asserted in the main run)."""
    first, last = band.flat_columns
    flat_w = w[first : last + 1]
    flat_m = multipliers[first : last + 1]
    column = apex_column(band)
    energy = float(np.sum(column**2))
    jitter_sd, jitter_energy = {}, {}
    for sigma in JITTER_PERIODS:
        delays = jitter_delays(w, sigma, period)[first : last + 1]
        jitter_sd[sigma] = abs(float(delays.std()) / (sigma * period) - 1.0)
        shifted = time_jitter(column, w, sigma, period)
        jitter_energy[sigma] = float(
            np.max(np.abs(np.sum(shifted**2, axis=0) / energy - 1.0))
        )
    return TextureCheck(
        w_mean=abs(float(flat_w.mean())),
        w_sd=abs(float(flat_w.std()) - 1.0),
        amplitude_mean_square=abs(float(np.mean(flat_m**2)) - 1.0),
        jitter_sd=jitter_sd,
        jitter_energy=jitter_energy,
    )


def texture_rng(position_id: str) -> np.random.Generator:
    """Generator of the shared noise field of one position (§2-7)."""
    return np.random.default_rng([TEXTURE_SEED, _crc(position_id)])


# ------------------------------------------------------------- injection


def inject_bands(
    context: inject.LineContext,
    placements: Sequence[tuple[inject.InjectionSpec, inject.BandPatch]],
) -> inject.BandInjectionResult:
    """``inject.inject_band_realization`` for bands that are already built.

    Same order and rules: zones (a)-(d), the band spacing (e), ``sigma`` over
    apex +- 1 P x the flat traces of the background, the int16 rule. The
    shifted and textured bands cannot be rebuilt by the library from a
    table row, so they are passed in (``inject`` private helpers keep the
    arithmetic identical; ``src`` is sealed and gets no new public call).

    Raises:
        ValueError: If the rows mix realizations, lines or modes.
    """
    specs = [spec for spec, _ in placements]
    realization_id = inject._check_one_realization(
        specs, context.background.line_id, inject.InjectionMode.BAND
    )
    data = np.asarray(context.background.data, dtype=np.float64).copy()
    w_eff = inject.effective_median_window(context)
    n_samples = context.background.n_samples
    placed: list[tuple[float, int]] = []
    truths: list[inject.BandTruth] = []
    rejected: list[tuple[str, str]] = []
    for spec, band in placements:
        failure = inject.band_placement_failure(context, band, spec.apex_trace)
        if failure is None and not inject.band_spacing_ok(
            spec.apex_trace, band.length_traces, placed, w_eff
        ):
            failure = inject.RejectReason.BAND_SPACING
        if failure is not None:
            rejected.append((spec.injection_id, failure.value))
            continue
        offset = inject.band_offset(band, spec.apex_trace)
        assert offset is not None  # checked by band_placement_failure
        scale = inject._band_scale(spec, context, band, offset)
        if isinstance(scale, inject.RejectReason):
            rejected.append((spec.injection_id, scale.value))
            continue
        protected = (
            inject.band_truth_box(band, spec.apex_trace, n_samples),
            inject.band_snr_window(context, band, offset),
        )
        summed, clip = inject._clip_field_sum(
            data, band.field, band.signal_amplitude, offset, scale.gain, protected
        )
        if clip.rejects:
            overflow = inject.RejectReason.INT16_OVERFLOW.value
            rejected.append((spec.injection_id, overflow))
            continue
        data[:, offset : offset + band.n_columns] = summed
        placed.append((spec.apex_trace, band.length_traces))
        truths.append(inject._make_band_truth(spec, context, band, scale, clip))
    return inject.BandInjectionResult(
        realization_id=realization_id,
        line_id=context.background.line_id,
        data=data,
        truths=tuple(truths),
        rejected=tuple(rejected),
    )


OFF_INT16_REJECTION = "int16_overflow_untapered"
"""An off-arm field rejected by the int16 rule (plan §2-3: the pair drops)."""


@dataclass(frozen=True, eq=False)
class UntaperedInjection:
    """The off arm of one hyperbola realization.

    Attributes:
        data: float64 line plus the accepted untapered fields.
        accepted: Injection ids whose off field was added.
        rejected: ``(injection_id, reason)`` of the off fields left out.
        n_clipped_cells: Clipped cells per accepted injection id.
    """

    data: np.ndarray
    accepted: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    n_clipped_cells: Mapping[str, int]


def inject_untapered(
    context: inject.LineContext,
    on_result: inject.InjectionResult,
    patches: Mapping[tuple[str, float], inject.ResponsePatch],
) -> UntaperedInjection:
    """The off arm: the on-arm gains on the untapered fields (plan §2-3).

    The on arm (``inject.inject_realization``) fixes acceptance, gain and
    truth; the off line is rebuilt from the background with the same gain
    on each untapered field, in the same order. The int16 rule is applied
    again on the off sum (truth box and SNR window protected, 5 % ratio);
    an off field it rejects is left out and its pair drops.

    Args:
        context: The background line.
        on_result: The on-arm realization.
        patches: ``(response_id, k) -> ResponsePatch`` used by the on arm.
    """
    data = np.asarray(context.background.data, dtype=np.float64).copy()
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    clipped: dict[str, int] = {}
    for truth in on_result.truths:
        patch = patches[(truth.response_id, truth.k)]
        offset = round(truth.apex_trace - patch.response.apex_trace)
        protected = (truth.box, inject.snr_window(context, patch, truth.apex_trace))
        summed, clip = inject._clip_field_sum(
            data,
            untapered_field(patch),
            patch.signal_amplitude,
            offset,
            truth.gain,
            protected,
        )
        if clip.rejects:
            rejected.append((truth.injection_id, OFF_INT16_REJECTION))
            continue
        data[:, offset : offset + patch.response.n_traces] = summed
        accepted.append(truth.injection_id)
        clipped[truth.injection_id] = clip.n_clipped_cells
    return UntaperedInjection(data, tuple(accepted), tuple(rejected), clipped)


def aperture_taper_columns(
    patches: Sequence[tuple[inject.ResponsePatch, float]], n_traces: int
) -> np.ndarray:
    """Line columns inside an aperture taper of any injection (precheck 4)."""
    inside = np.zeros(n_traces, dtype=bool)
    for patch, apex_trace in patches:
        offset = round(apex_trace - patch.response.apex_trace)
        n_taper = patch.n_aperture_taper
        width = patch.response.n_traces
        inside[max(0, offset) : max(0, offset + n_taper)] = True
        inside[max(0, offset + width - n_taper) : max(0, offset + width)] = True
    return inside


def off_equals_on_outside_tapers(
    on_data: np.ndarray, off_data: np.ndarray, taper_columns: np.ndarray
) -> bool:
    """Precheck 4: bit-for-bit equal outside every aperture-taper column."""
    keep = ~taper_columns
    return bool(np.array_equal(on_data[:, keep], off_data[:, keep]))


def aperture_end_traces(
    patch: inject.ResponsePatch, apex_trace: float
) -> tuple[int, int]:
    """First and last line trace of the injected aperture (the abrupt ends)."""
    offset = round(apex_trace - patch.response.apex_trace)
    return offset, offset + patch.response.n_traces - 1


def boxes_near_traces(
    boxes: Sequence[GridBox], traces: Sequence[int], margin_traces: float
) -> int:
    """Boxes whose trace span comes within ``margin`` of any given trace."""
    count = 0
    for box in boxes:
        if any(box.i0 - margin_traces <= t <= box.i1 + margin_traces for t in traces):
            count += 1
    return count


# --------------------------------------------------------------- contrast


def centred_grey(image_values: np.ndarray, content_height_px: int) -> np.ndarray:
    """``q = grey - median grey of the content`` (the contrast script's input)."""
    content = image_values[:content_height_px]
    return image_values.astype(np.float64) - float(np.median(content))


def _rms(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _db(ratio: float) -> float:
    return 20 * math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else math.nan


def band_contrast(
    q: np.ndarray,
    box_px: tuple[float, float, float, float],
    h_limit: int,
    side_gap: int = 0,
) -> dict | None:
    """``c_vert_db``, ``c_lat_db`` and ``coh`` of a box ``[x1, x2) x [y1, y2)``.

    The definition of ``m33_r3_band_contrast.metrics`` ([R2R] §1-2): the box
    centre half against two box heights above and below (vertical) and one
    box width beside it outside ``side_gap`` (lateral); ``coh`` is the mean
    over the box rows of the squared row mean over the mean square.

    Returns:
        The three values with ``rms_in``, ``w_px`` and ``h_px``, or ``None``
        for a box narrower than 4 px or lower than 2 px.
    """
    height, width = q.shape
    h_limit = min(h_limit, height)
    x1, y1, x2, y2 = box_px
    x1, x2 = max(0, math.floor(x1)), min(width, math.ceil(x2))
    y1, y2 = max(0, math.floor(y1)), min(h_limit, math.ceil(y2))
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 4 or box_h < 2:
        return None
    c0, c1 = x1 + box_w // 4, x2 - box_w // 4
    inside = q[y1:y2, c0:c1]
    above = q[max(0, y1 - 2 * box_h) : y1, c0:c1]
    below = q[y2 : min(h_limit, y2 + 2 * box_h), c0:c1]
    vertical = np.concatenate([above.ravel(), below.ravel()])
    left = q[y1:y2, max(0, x1 - side_gap - box_w) : max(0, x1 - side_gap)]
    right = q[y1:y2, min(width, x2 + side_gap) : min(width, x2 + side_gap + box_w)]
    lateral = np.concatenate([left.ravel(), right.ravel()])
    box = q[y1:y2, x1:x2].astype(np.float64)
    coh = float(np.mean(box.mean(axis=1) ** 2) / max(np.mean(box**2), 1e-12))
    rms_in = _rms(inside)
    return {
        "c_vert_db": _db(rms_in / _rms(vertical)) if vertical.size >= 8 else math.nan,
        "c_lat_db": _db(rms_in / _rms(lateral)) if lateral.size >= 8 else math.nan,
        "coh": coh,
        "rms_in": rms_in,
        "w_px": box_w,
        "h_px": box_h,
    }


def truth_box_px(
    box: GridBox, traces_per_px: float, samples_per_px: float
) -> tuple[float, float, float, float]:
    """L1 px rectangle of an L0 truth box on an image (cell edges)."""
    return (
        box.i0 / traces_per_px,
        box.k0 / samples_per_px,
        (box.i1 + 1) / traces_per_px,
        (box.k1 + 1) / samples_per_px,
    )
