"""4TU depth validation (CHECKLIST M2-8, spec ``docs/processing/spec_eval_4tu_v1.md``).

For every candidate (``fourtu_apex_candidates_v1.csv``) a box is supplied (spec 4-1) and
``estimate_depth`` runs with **default settings** (no 4TU tuning; the zero is
computed once per line). Depths are clustered per activity across parallel
lines (4-3, v1.3 complete linkage of width W), matched to the ground truth with
an ungated Hungarian assignment (5), compared with a random-depth baseline (6),
and written to ``docs/processing/<prefix>_*.csv`` and ``<prefix>.md`` (8).

Variants reported (spec 5):

* assumption: ``top`` (gt = ``depth_m``) or ``center`` (gt = ``depth_m - R``)
* zero_variant: ``detected`` (main), ``est`` / ``act`` (metadata zeros)
* variant: ``main`` (clusters seen on >= 2 parallel lines), ``a`` (clusters,
  no support filter), ``b`` (every candidate depth, no clustering); these three
  drop the candidates excluded by the main apex-candidate QC rule B+D+E
  (3-5-5, v1.5: curvature out or unknown, record end, trace seam) before
  clustering. ``bd`` drops only the B+D candidates (curvature out, record end,
  trace seam; the v1.4.4 main result) and then clusters and filters like
  ``main``. ``noqc`` is ``main`` without the QC (the v1.3.1 main result).
* gt_set: ``exclude_x`` (main, code X dropped) or ``include_x``; both come
  from the same run and differ only in the matching step

Truth-dependent code stays here, not in ``src``. ``--dry-run`` only parses the
candidate, line and ground-truth tables (no radargram is loaded).

Example:
    conda run -n gpr-dev python -I scripts/eval_depth_4tu.py --dry-run
    conda run -n gpr-dev python -I scripts/eval_depth_4tu.py
"""

import argparse
import csv
import itertools
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.depth.estimate import (
    DepthInputs,
    estimate_depth,
    prepare_depth_inputs,
)
from gpr_detection.depth.hyperbola import (
    V_MAX_M_PER_NS,
    V_MIN_M_PER_NS,
    CurvatureFit,
    FitStatus,
    HyperbolaFit,
    fit_curvature,
)
from gpr_detection.depth.picks import (
    APEX_DESCENDED,
    ApexHint,
    HyperbolaPicks,
    TrackStop,
    extract_picks,
    mute_sample,
)
from gpr_detection.depth.result import DepthResult, DepthStatus
from gpr_detection.io.fourtu import (
    FourTUTable,
    T0Choice,
    load_fourtu_radargram,
    load_fourtu_table,
    site_of_activity,
)
from gpr_detection.io.paths import load_dataset_paths
from gpr_detection.io.radargram import GridBox, Radargram, velocity_from_eps
from gpr_detection.preprocess.filters import period_ns_or_default
from gpr_detection.preprocess.timezero import DEFAULT_PERIOD_NS, ZeroResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APEX_CANDIDATES = (
    PROJECT_ROOT / "docs" / "processing" / "fourtu_apex_candidates_v1.csv"
)
DEFAULT_LINES = PROJECT_ROOT / "docs" / "processing" / "fourtu_lines_v1.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "processing"
DEFAULT_PREFIX = "depth_4tu_v1"

BOX_HALF_WIDTH_M = 0.8
BOX_MARGIN_PERIODS = 1.5
"""Candidate box (spec 4-1): +-0.8 m, 1.5 periods above it and below the wing."""

CANDIDATE_MOVED_TRACES = 3
CANDIDATE_MOVED_PERIODS = 0.5
"""``candidate_moved`` when the apex is > 3 traces or > 0.5 / f_c from the candidate."""

CLUSTER_WIDTH_M = 0.10
"""Cluster width W (spec 4-4): every member pair of a cluster lies within W."""
CLUSTER_EDGE_TOL_M = 1e-9
"""A merged width of W + 1e-9 m still merges (boundary included, spec 4-3-1)."""
CLUSTER_TIE_TOL_M = 1e-12
"""Merged widths within 1e-12 m are a tie; the shallower pair wins (4-3-1)."""
CLUSTER_LINKAGES = ("complete", "single")
"""``single`` keeps the v1.2 chaining behaviour for the B1 test and diagnostics."""
CLUSTER_SPAN_DECIMALS = 6
"""``span_m`` in the clusters csv: micrometres, fine enough to see a width breach."""
SPAN_QUANTILES_PERCENT = (0, 50, 95, 100)
"""``span_m`` quantiles in the md cluster diagnostics (spec 9-1)."""
MIN_SUPPORT_LINES = 2
PASS_LIMIT_M = 0.30
"""Article 186 tolerance: |e| <= 0.30 m passes (boundary included)."""

BASELINE_TRIALS = 1000
BASELINE_SEED = 0
SINGLE_GT_HALF_RANGE_M = 0.3
GLOBAL_BASELINE_RANGE_M = (0.2, 2.0)
"""Secondary baseline range (assumed, spec 6)."""

LONG_LINE_M = 30.0
SITE_LONG_DT = 11
SITE_TWO_EVENTS = 9
SITE_11_DIAGNOSTIC_EPS = 8.0
"""Site 11 diagnostic only: depth scaled to eps_r 8 (never used for judging)."""

ZERO_VARIANTS = ("detected", "est", "act")
META_ZERO_KEYS = {"est": "t0_sample_est", "act": "t0_sample_act_median"}
ASSUMPTIONS = ("top", "center")
VARIANTS = ("main", "a", "b", "bd", "noqc")
"""``bd`` (v1.5): ``main`` with the B+D rule; ``noqc`` (v1.4): ``main`` without the
apex-candidate QC. Both only exist with ``--apex-qc on`` (spec 5)."""
BD_VARIANT = "bd"
NO_QC_VARIANT = "noqc"
QC_AUX_VARIANTS = (BD_VARIANT, NO_QC_VARIANT)
"""Auxiliary QC variants, dropped with ``--apex-qc off`` (3-5-5 item 7)."""
COUNTED_VARIANTS = ("main", *QC_AUX_VARIANTS)
"""Variants whose detected-zero estimate counts feed md section 8 (v1.5)."""
CODE_NOT_A_PIPE = "X"
SITE11_VARIANTS = ("main_path", "unclustered")
"""Site 11 diagnostic: clustered + support filter, or every candidate."""
ORDER_MATCH_COLUMN = "정답표와 순서 일치"
ORDER_MATCH_YES = "예"
BASELINE_TRIALS_AUX = 200
"""Trials for the auxiliary variants (a, b, bd, noqc) and the global baseline (6)."""

# Apex-candidate QC (spec 3-5, v1.4). Values are locked in
# docs/processing/m29c_qc_constants.json; they are not CLI options.
QC_R_LO = 0.7
QC_R_HI = 1.6
QC_R_TOL = 1e-12
"""Curvature ratio band ``0.7 - tol <= r <= 1.6 + tol`` is ``in`` (3-5-2)."""
QC_UNSTABLE_SIGMA_REL = 0.15
QC_UNSTABLE_INLIER = 0.5
"""``unstable`` when ``inlier_frac < 0.5`` or ``sigma_v / v_m > 0.15``."""
QC_BOUND_REL = 1e-4  # spec_depth 6-3 boundary rule
QC_RADIUS_M = 0.0
"""The QC fit uses R = 0: the pipe radius of a candidate is unknown (3-5-2)."""
RECORD_END_PERIODS = 2.5
RECORD_END_TOL_NS = 1e-9
"""``apex_at_record_end`` when less than 2.5 / f_c - 1e-9 ns is left below (3-5-3)."""
SEAM_TIME_PERIODS = 0.1
SEAM_AMP_RATIO = 1.25
SEAM_NOISE_K = 5.0
"""Seam score limits: ``max(0.1 / f_c, 5 sigma)`` and ``max(ln 1.25, 5 sigma)``."""
SEAM_ZERO_HALF_PERIODS = 1.0
SEAM_LAG_HALF_PERIODS = 0.5
"""Zero-band window ``t_z +- 1 / f_c``; lags within ``+- 0.5 / f_c`` (3-5-4)."""
SEAM_REF_HALF_M = BOX_HALF_WIDTH_M
SEAM_APEX_HALF_M = 0.10
"""Noise reference over +-0.8 m of splits; seams within +-0.10 m of the apex."""
SEAM_SCORE_LIMIT = 1.0
"""A split with ``s_j >= 1`` is a seam (3-5-4 item 6)."""
QC_FLAG_OUT = "curvature_v_out"
QC_FLAG_UNKNOWN = "curvature_v_unknown"
FLAG_RECORD_END = "apex_at_record_end"
FLAG_SEAM = "apex_on_trace_seam"
MAD_TO_SIGMA = 1.4826
TRACE_COUNT_TOL = 1e-9
"""Trace counts ``ceil(half / dx - 1e-9)``: a quotient a hair above an integer
must not add a trace (same guard as ``candidate_box``)."""
QC_R_QUANTILES_PERCENT = (0, 10, 50, 90, 100)
"""Quantiles of the known curvature ratios in md section 8 (spec 8)."""
# The spec's en dashes and U+2212 minus are kept verbatim (test C20), hence RUF001.
PREREGISTERED_QC_NOTE = (
    '"QC는 후보 단위로 묶기 전에 적용한다. 주 결과(B+D+E)는 곡률 비가 띠 '
    "0.7–1.6 밖인 후보(B), 곡률 v를 구할 수 없는 후보(E), 기록 끝·트레이스 "  # noqa: RUF001
    "이음에 걸린 후보(D)를 뺀다. E는 날개 추적 개선(spec_depth v1.9) 뒤, "
    "재평가 전에 사용자가 채택한 규칙이다. 채택 근거는 정답 깊이를 가린 수동 "
    "검토 라벨(판독자 1명, 검토 표본 169개 = 수동 39 + 자동 130)로 센 효과다: "
    "B+D 대비 자동 후보 정밀도 14.3 %(95 % 7.8–23.2 %) → 35.7 %(18.6–55.9 %), "  # noqa: RUF001
    "진짜 꼭짓점 남김 39/41 → 29/41(자동 13/14 → 10/14), 남은 후보 126 → "
    "49(−61 %). 이 수치는 목표가 아니라 추정이다. E는 새 문턱을 만들지 않았고, "  # noqa: RUF001
    "4TU 정답 깊이와 M2-8 결과를 보지 않고 정했다. E는 추정 수를 크게 줄이므로 "
    "누락이 늘어 정답 기준 판정률 B가 내려갈 수 있고, 추정 수를 같게 맞춘 무작위 "
    '기준선도 함께 바뀐다(설계상 성질)." 곡률 unknown을 남긴 B+D 결과(`bd`)와 '
    "QC 없는 결과(`noqc`)를 보조로 함께 보고한다."
)
"""Pre-registration text of spec section 11 item 3 (v1.5), quoted verbatim in md
section 8 (test C20 compares it with the spec)."""
QC_COLUMNS = (
    "qc_verdict",
    "qc_reason",
    "qc_r",
    "qc_v_curv_m_per_ns",
    "qc_sigma_v_rel",
    "qc_inlier_frac",
    "qc_n_picks",
    "qc_n_left",
    "qc_n_right",
    "qc_aperture_m",
    "record_end_margin_ns",
    "apex_at_record_end",
    "seam_score",
    "seam_split",
    "apex_on_trace_seam",
    "qc_excluded",  # main rule B+D+E (v1.5); B+D = out or record end or seam
)
"""QC columns of the candidates csv, after ``apex_minus_search_k0`` (spec 8)."""
TRACKING_COLUMNS = ("stop_left", "stop_right", "apex_descended")
"""Tracking diagnostic columns after ``qc_excluded`` (spec_depth v1.9, 13-4).

``stop_*`` come from the QC picks (blank with ``--apex-qc off``);
``apex_descended`` comes from ``DepthResult.flags``. Never used by a verdict."""
LOGGER = logging.getLogger("eval_depth_4tu")


# ----------------------------------------------------------------- data classes


@dataclass(frozen=True)
class GtItem:
    """One ground-truth utility of an activity (spec 2)."""

    depth_m: float
    code: str
    radius_m: float
    radius_known: bool
    trench_no: str
    multi_trench: bool

    def depth_for(self, assumption: str) -> float:
        """``top``: ``depth_m``; ``center``: ``depth_m - R`` (R 0 when unknown)."""
        return self.depth_m if assumption == "top" else self.depth_m - self.radius_m


class QcVerdict(StrEnum):
    """Curvature QC verdict of one candidate (spec 3-5-2)."""

    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CurvatureQc:
    """Curvature QC of one candidate (rule B, spec 3-5-2).

    There is no depth field on purpose (3-5-6): the QC fit never feeds a depth.

    Attributes:
        verdict: ``in``, ``out`` or ``unknown``.
        reason: ``""`` for ``in``; ``r_below``, ``r_above``, ``v_at_upper_bound``,
            ``v_at_lower_bound`` for ``out``; ``no_box``, ``no_apex``,
            ``insufficient_picks``, ``fit_failed``, ``unstable`` for ``unknown``.
        r: ``v_curv / v_m``; ``None`` without a converged fit.
        v_curv_m_per_ns: R = 0 straight-ray curvature velocity (QC only).
        sigma_v_rel: ``sigma_v / v_m``; ``None`` when not finite.
        inlier_frac: Inlier fraction of the QC fit.
        n_picks, n_left, n_right: Pick counts of the depth path's picks.
        aperture_m: ``max |x - x_apex|`` over the picks.
        stop_left, stop_right: Why each wing of those picks ended
            (diagnostic only, spec_depth v1.9 13-4).
    """

    verdict: QcVerdict
    reason: str
    r: float | None = None
    v_curv_m_per_ns: float | None = None
    sigma_v_rel: float | None = None
    inlier_frac: float | None = None
    n_picks: int = 0
    n_left: int = 0
    n_right: int = 0
    aperture_m: float | None = None
    stop_left: TrackStop | None = None
    stop_right: TrackStop | None = None


@dataclass(frozen=True)
class CandidateQc:
    """Apex-candidate QC: curvature (B), record end and trace seam (D), spec 3-5."""

    curvature: CurvatureQc
    record_end_margin_ns: float
    at_record_end: bool
    seam_score: float | None
    """Largest seam score near the apex; ``None`` without a zero."""
    seam_split: int | None
    """Split ``j`` (between traces ``j - 1`` and ``j``) of ``seam_score``."""
    on_trace_seam: bool

    @property
    def excluded_bd(self) -> bool:
        """B+D rule (spec 3-5-5, v1.4 main rule; the ``bd`` variant)."""
        return (
            self.curvature.verdict == QcVerdict.OUT
            or self.at_record_end
            or self.on_trace_seam
        )

    @property
    def excluded(self) -> bool:
        """B+D+E rule: dropped from the main result before clustering (3-5-5 item 1,
        v1.5).

        Every curvature verdict other than ``in`` is excluded, so an unknown
        reason added later is excluded without touching this rule.
        """
        return (
            self.curvature.verdict != QcVerdict.IN
            or self.at_record_end
            or self.on_trace_seam
        )

    @property
    def flags(self) -> tuple[str, ...]:
        """QC flags for the candidates csv: curvature, record end, seam."""
        flags = []
        if self.curvature.verdict == QcVerdict.OUT:
            flags.append(QC_FLAG_OUT)
        elif self.curvature.verdict == QcVerdict.UNKNOWN:
            flags.append(QC_FLAG_UNKNOWN)
        if self.at_record_end:
            flags.append(FLAG_RECORD_END)
        if self.on_trace_seam:
            flags.append(FLAG_SEAM)
        return tuple(flags)


@dataclass(frozen=True)
class CandidateResult:
    """One candidate and its ``estimate_depth`` result (spec 4-2)."""

    activity_id: str
    path: str
    candidate_trace: int
    candidate_sample: float
    source: str
    result: DepthResult
    candidate_moved: bool
    box_k0: int | None = None
    """Top of the (widened) box; ``None`` without a box."""
    search_k0: int | None = None
    """Search top ``k_top``: the later of ``box_k0`` and the mute (5-7 item 2)."""
    qc: CandidateQc | None = None
    """Apex-candidate QC (3-5); ``None`` with ``--apex-qc off``."""

    def depth(self, zero_variant: str) -> float | None:
        """Pipe-top depth for the detected zero or a metadata zero (never reads qc)."""
        if self.result.status != "ok":
            return None
        if zero_variant == "detected":
            return self.result.depth_top_m
        alternative = self.result.alt_depths.get(META_ZERO_KEYS[zero_variant])
        return None if alternative is None else alternative.depth_top_m


@dataclass(frozen=True)
class Cluster:
    """Depth cluster across the lines of one activity (4-3)."""

    depth_m: float
    """Median of the member depths."""
    n_lines: int
    n_apex_candidates: int
    span_m: float
    """Deepest minus shallowest member depth."""
    member_indices: tuple[int, ...] = ()
    """Indices of the members in the input ``depths`` sequence, in sort-key order
    ``(depth, str(line), index)``."""


@dataclass(frozen=True)
class MatchResult:
    """Ungated Hungarian matching of one activity (spec 5)."""

    pairs: tuple[tuple[float, int, float], ...]
    """``(estimate, gt index, error = estimate - gt)`` per pair."""
    missed: int
    false_est: int
    est_indices: tuple[int, ...] = ()
    """Index into ``est`` of each pair (same order as ``pairs``)."""


@dataclass(frozen=True)
class BaselineStats:
    """Random-depth baseline over ``trials`` (spec 6)."""

    mae_mean: float
    mae_p025: float
    mae_p975: float
    pass_a_mean: float
    pass_b_mean: float
    pass_b_p025: float
    pass_b_p975: float
    pass_a_p025: float
    pass_a_p975: float
    pass_b_values: np.ndarray = field(repr=False)


# ----------------------------------------------------------------- candidates and boxes


def candidate_box(
    radargram: Radargram, trace: int, sample: float, tau_ns: float
) -> GridBox:
    """Apex candidate -> box (spec 4-1), wing time from the metadata velocity.

    Args:
        radargram: The 4TU line (needs ``dt_ns``, ``dx_m``, ``f_c_hz``,
            ``eps_r_meta``).
        trace: Candidate trace (L0).
        sample: Candidate sample (L0, fractional).
        tau_ns: Candidate time after the detected zero (``tau_ns`` column).
    """
    assert radargram.dt_ns is not None and radargram.dx_m is not None
    assert radargram.f_c_hz is not None and radargram.eps_r_meta is not None
    dt_ns = radargram.dt_ns
    period_ns = 1e9 / radargram.f_c_hz
    half_width = math.ceil(BOX_HALF_WIDTH_M / radargram.dx_m - 1e-9)
    v_meta = velocity_from_eps(radargram.eps_r_meta)
    candidate_depth_m = v_meta * tau_ns / 2.0
    wing_delay_ns = (2.0 / v_meta) * (
        math.hypot(candidate_depth_m, BOX_HALF_WIDTH_M) - candidate_depth_m
    )
    k0 = math.floor(sample - BOX_MARGIN_PERIODS * period_ns / dt_ns)
    k1 = math.ceil(sample + (wing_delay_ns + BOX_MARGIN_PERIODS * period_ns) / dt_ns)
    return GridBox(
        i0=max(0, trace - half_width),
        i1=min(radargram.n_traces - 1, trace + half_width),
        k0=max(0, k0),
        k1=min(radargram.n_samples - 1, k1),
    )


def _candidate_moved(
    radargram: Radargram, trace: int, sample: float, result: DepthResult
) -> bool:
    if result.apex_trace is None or result.apex_sample is None:
        return False
    assert radargram.dt_ns is not None and radargram.f_c_hz is not None
    sample_limit = CANDIDATE_MOVED_PERIODS * 1e9 / radargram.f_c_hz / radargram.dt_ns
    return (
        abs(result.apex_trace - trace) > CANDIDATE_MOVED_TRACES
        or abs(result.apex_sample - sample) > sample_limit
    )


def _search_top(radargram: Radargram, box: GridBox, zero: ZeroResult) -> int:
    """``k_top`` of spec_depth_v1 5-7: widened box top or the mute, the later."""
    if radargram.dt_ns is None or zero.t_zero_ns is None:
        return box.k0
    period_ns = period_ns_or_default(radargram.f_c_hz, DEFAULT_PERIOD_NS)
    return max(box.k0, mute_sample(zero.t_zero_ns, period_ns, radargram.dt_ns))


# ----------------------------------------------------------------- apex candidate QC
# (spec 3-5, v1.4)
#
# The QC only decides whether a candidate enters the main result. Its curvature
# fit never reaches a depth, an uncertainty or a ``DepthResult`` (3-5-6).


def _traces_within(half_width_m: float, dx_m: float) -> int:
    """``ceil(half_width / dx)`` without the float quotient's spurious extra trace."""
    return math.ceil(half_width_m / dx_m - TRACE_COUNT_TOL)


def classify_ratio(r: float) -> QcVerdict:
    """Rows 9-11 of the 3-5-2 table: ``in`` inside ``[0.7, 1.6]`` (+-1e-12).

    Raises:
        ValueError: ``r`` is NaN or infinite.
    """
    if not math.isfinite(r):
        raise ValueError(f"curvature ratio must be finite, got {r}")
    if QC_R_LO - QC_R_TOL <= r <= QC_R_HI + QC_R_TOL:
        return QcVerdict.IN
    return QcVerdict.OUT


def _near_bound(value: float, low: float, high: float) -> bool:
    """Within ``QC_BOUND_REL`` of the range from either bound, or outside it."""
    return min(value - low, high - value) <= QC_BOUND_REL * (high - low)


def _fit_verdict(
    fit: HyperbolaFit, picks: HyperbolaPicks, v_meta: float
) -> tuple[QcVerdict, str]:
    """Rows 5-11 of the 3-5-2 table for a converged fit (first matching row wins).

    The velocity-bound rows (6, 7) come before the stability row (8): a flat
    event fits no hyperbola well, so its inlier fraction is low, yet the bound
    already says it lies outside the band.
    """
    x_m, tau_ns = picks.x_m, picks.tau_ns
    if _near_bound(fit.x0_m, float(x_m.min()), float(x_m.max())) or _near_bound(
        fit.t0_ns, 0.0, float(tau_ns.max())
    ):
        return QcVerdict.UNKNOWN, "fit_failed"
    v_bound_margin = QC_BOUND_REL * (V_MAX_M_PER_NS - V_MIN_M_PER_NS)
    if V_MAX_M_PER_NS - fit.v_m_per_ns <= v_bound_margin:
        return QcVerdict.OUT, "v_at_upper_bound"
    if fit.v_m_per_ns - V_MIN_M_PER_NS <= v_bound_margin:
        return QcVerdict.OUT, "v_at_lower_bound"
    is_unstable = (
        fit.inlier_frac < QC_UNSTABLE_INLIER
        or not math.isfinite(fit.sigma_v)
        or fit.sigma_v / v_meta > QC_UNSTABLE_SIGMA_REL
    )
    if is_unstable:
        return QcVerdict.UNKNOWN, "unstable"
    r = fit.v_m_per_ns / v_meta
    if classify_ratio(r) == QcVerdict.IN:
        return QcVerdict.IN, ""
    return QcVerdict.OUT, "r_below" if r < QC_R_LO - QC_R_TOL else "r_above"


def classify_curvature(
    curvature: CurvatureFit, picks: HyperbolaPicks, v_meta: float
) -> CurvatureQc:
    """Curvature verdict from a QC fit (rows 2-11 of the 3-5-2 table, pure).

    Args:
        curvature: ``fit_curvature`` result of the QC (R = 0, ``"ground"``).
        picks: The picks that were fitted (the depth path's picks).
        v_meta: Metadata velocity ``v_m`` in m/ns.
    """
    if picks.apex is None:
        return _unknown_curvature("no_apex", picks)
    if curvature.status == FitStatus.INSUFFICIENT_PICKS:
        return _unknown_curvature("insufficient_picks", picks)
    fit = curvature.fit
    if fit is None or not fit.converged:
        return _unknown_curvature("fit_failed", picks)
    verdict, reason = _fit_verdict(fit, picks, v_meta)
    sigma_v_rel = fit.sigma_v / v_meta
    return CurvatureQc(
        verdict,
        reason,
        r=fit.v_m_per_ns / v_meta,
        v_curv_m_per_ns=fit.v_m_per_ns,
        sigma_v_rel=sigma_v_rel if math.isfinite(sigma_v_rel) else None,
        inlier_frac=fit.inlier_frac,
        aperture_m=float(np.max(np.abs(picks.x_m - picks.apex_x_m))),
        n_picks=picks.n_picks,
        n_left=picks.n_left,
        n_right=picks.n_right,
    )


def _unknown_curvature(reason: str, picks: HyperbolaPicks) -> CurvatureQc:
    return CurvatureQc(
        QcVerdict.UNKNOWN,
        reason,
        n_picks=picks.n_picks,
        n_left=picks.n_left,
        n_right=picks.n_right,
    )


def _check_same_apex(picks: HyperbolaPicks, result: DepthResult) -> None:
    """The QC must see the depth path's apex (3-5-2 picks); else the contract broke."""
    if picks.apex is None:
        return
    qc_apex = (picks.apex.trace, picks.apex.k)
    depth_apex = (result.apex_trace, result.apex_sample)
    if qc_apex != depth_apex:
        raise RuntimeError(
            f"QC picks apex {qc_apex} differs from the depth result apex {depth_apex}"
        )


def _qc_meta_velocity(radargram: Radargram, result: DepthResult) -> float:
    """``v_m = c / sqrt(eps_r_used)``; the metadata eps_r when the depth has none."""
    eps_r = result.eps_r_used if result.eps_r_used is not None else radargram.eps_r_meta
    assert eps_r is not None, "candidate_box already needs eps_r_meta"
    return velocity_from_eps(eps_r)


def curvature_qc(
    radargram: Radargram,
    box: GridBox,
    inputs: DepthInputs,
    hint: ApexHint,
    result: DepthResult,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> CurvatureQc:
    """Curvature QC of one candidate after its depth (rule B, spec 3-5-2).

    Re-extracts the picks with exactly the arguments ``estimate_depth`` used
    (the code is deterministic, so they are the depth path's picks) and fits
    ``R = 0`` with the ``"ground"`` label. The radargram is not changed and the
    fit result goes nowhere but the verdict (3-5-6).

    Raises:
        RuntimeError: The picks' apex differs from ``result``'s apex.
    """
    picks = extract_picks(
        inputs.branch.background_removed,
        box,
        inputs.zero,
        dt_ns=radargram.dt_ns,
        dx_m=radargram.dx_m,
        f_c_hz=radargram.f_c_hz,
        config=config,
        apex_hint=hint,
        period_px=inputs.branch.period_px,
    )
    _check_same_apex(picks, result)
    if picks.apex is not None and inputs.zero.t_zero_ns is None:
        # Not reachable for 4TU candidates (no zero means no tau_ns, so no box);
        # without a zero the picks have no time axis to fit.
        return CurvatureQc(
            QcVerdict.UNKNOWN,
            "fit_failed",
            n_picks=picks.n_picks,
            n_left=picks.n_left,
            n_right=picks.n_right,
            stop_left=picks.stop_left,
            stop_right=picks.stop_right,
        )
    v_meta = _qc_meta_velocity(radargram, result)
    curvature = fit_curvature(
        picks,
        radius_m=QC_RADIUS_M,
        air_gap="ground",  # QC-only label; the depth path stays meta_fixed
        v_fallback=v_meta,
        config=config,
    )
    verdict = classify_curvature(curvature, picks, v_meta)
    return replace(verdict, stop_left=picks.stop_left, stop_right=picks.stop_right)


def record_end_margin_ns(sample: float, n_samples: int, dt_ns: float) -> float:
    """Time left from the candidate sample to the last sample, ``(N - 1 - k) dt``."""
    return (n_samples - 1 - sample) * dt_ns


def apex_at_record_end(
    sample: float, n_samples: int, dt_ns: float, f_c_hz: float
) -> bool:
    """Rule D record end (3-5-3): less than ``2.5 / f_c`` left below the candidate."""
    end_zone_ns = RECORD_END_PERIODS * 1e9 / f_c_hz
    margin_ns = record_end_margin_ns(sample, n_samples, dt_ns)
    return margin_ns < end_zone_ns - RECORD_END_TOL_NS


def _zero_band_window(
    t_zero_ns: float, dt_ns: float, period_ns: float, n_samples: int, max_lag: int
) -> tuple[int, int]:
    """Samples ``[k0, k1]`` of ``t_z +- 1 / f_c``, ``max_lag`` inside the record."""
    half_ns = SEAM_ZERO_HALF_PERIODS * period_ns
    k0 = math.floor((t_zero_ns - half_ns) / dt_ns)
    k1 = math.ceil((t_zero_ns + half_ns) / dt_ns)
    return max(max_lag, k0), min(n_samples - 1 - max_lag, k1)


def _parabolic_offsets(correlations: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Sub-lag offset of each trace's maximum: inner and concave peaks only."""
    n_traces = correlations.shape[1]
    offsets = np.zeros(n_traces)
    columns = np.nonzero((best > 0) & (best < correlations.shape[0] - 1))[0]
    before = correlations[best[columns] - 1, columns]
    peak = correlations[best[columns], columns]
    after = correlations[best[columns] + 1, columns]
    curvature = before - 2 * peak + after
    is_concave = curvature < 0
    offsets[columns[is_concave]] = (
        0.5 * (before[is_concave] - after[is_concave]) / curvature[is_concave]
    )
    return offsets


def zero_band_track(
    filtered: np.ndarray, t_zero_ns: float, dt_ns: float, f_c_hz: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-trace zero-band lag (ns) and amplitude (3-5-4 items 1-3).

    The template is the per-sample median of all traces in ``t_z +- 1 / f_c``;
    every trace is cross-correlated with it over lags within ``+- 0.5 / f_c``.
    A template without energy (no zero band, v1.4.1) gives no reference event,
    so there is no track: zero amplitudes would make every split an ``inf`` seam.

    Args:
        filtered: Fitting-branch ``filtered`` section (signed, before the
            background removal) ``[n_samples, n_traces]``.
        t_zero_ns: Detected zero of the line.
        dt_ns: Sample interval.
        f_c_hz: Centre frequency.

    Returns:
        ``(lag_ns, amplitude)``: ``lag * dt`` with parabolic refinement and
        ``|c(lag*)| / sum(w^2)`` per trace. ``None`` when ``sum(w^2)`` is 0.
    """
    period_ns = 1e9 / f_c_hz
    data = np.asarray(filtered, dtype=float)
    max_lag = math.ceil(SEAM_LAG_HALF_PERIODS * period_ns / dt_ns)
    k0, k1 = _zero_band_window(t_zero_ns, dt_ns, period_ns, data.shape[0], max_lag)
    template = np.median(data[k0 : k1 + 1], axis=1)
    template_energy = float(template @ template)
    if template_energy <= 0:
        return None
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.empty((len(lags), data.shape[1]))
    for row, lag in enumerate(lags):
        correlations[row] = template @ data[k0 + lag : k1 + 1 + lag]
    best = np.argmax(correlations, axis=0)
    lag_ns = (lags[best] + _parabolic_offsets(correlations, best)) * dt_ns
    peak = correlations[best, np.arange(data.shape[1])]
    return lag_ns, np.abs(peak / template_energy)


def _robust_sigma(values: np.ndarray) -> float:
    """``1.4826 MAD`` of the finite values; 0 when there are none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return MAD_TO_SIGMA * float(np.median(np.abs(finite - np.median(finite))))


def _without_split(jumps: np.ndarray, split_index: int, half: int) -> np.ndarray:
    """Jumps of the splits ``1 <= |m - j| <= half`` around ``split_index``."""
    low, high = max(0, split_index - half), min(len(jumps), split_index + half + 1)
    return np.concatenate([jumps[low:split_index], jumps[split_index + 1 : high]])


def seam_scores(
    lag_ns: np.ndarray, amplitude: np.ndarray, f_c_hz: float, dx_m: float
) -> np.ndarray:
    """Seam score ``s_j`` per split between traces ``j - 1`` and ``j`` (3-5-4, pure).

    Returns:
        Array of ``n_traces - 1`` scores; index ``j - 1`` is split ``j``.
        ``s_j >= 1`` is a seam; a zero amplitude gives ``inf``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_jumps = np.diff(np.log(amplitude))
    gain_jumps[~np.isfinite(gain_jumps)] = np.inf
    lag_jumps = np.diff(lag_ns)
    time_floor_ns = SEAM_TIME_PERIODS * 1e9 / f_c_hz
    gain_floor = math.log(SEAM_AMP_RATIO)
    reference_half = _traces_within(SEAM_REF_HALF_M, dx_m)
    scores = np.empty(len(lag_jumps))
    for index in range(len(lag_jumps)):
        lag_noise = _robust_sigma(_without_split(lag_jumps, index, reference_half))
        gain_noise = _robust_sigma(_without_split(gain_jumps, index, reference_half))
        lag_limit = max(time_floor_ns, SEAM_NOISE_K * lag_noise)
        gain_limit = max(gain_floor, SEAM_NOISE_K * gain_noise)
        scores[index] = max(
            abs(lag_jumps[index]) / lag_limit, abs(gain_jumps[index]) / gain_limit
        )
    return scores


def seam_near(
    scores: np.ndarray, trace: int, dx_m: float
) -> tuple[float | None, int | None]:
    """Largest ``s_j`` with ``trace - J <= j - 1 and j <= trace + J`` (3-5-4 item 7).

    Returns:
        ``(score, j)``; ``(None, None)`` when no split is in range. Ties keep
        the first (shallowest-index) split.
    """
    half = _traces_within(SEAM_APEX_HALF_M, dx_m)
    best_score, best_split = None, None
    for split in range(max(1, trace - half + 1), min(len(scores), trace + half) + 1):
        score = float(scores[split - 1])
        if math.isnan(score):
            continue
        if best_score is None or score > best_score:
            best_score, best_split = score, split
    return best_score, best_split


def _line_seam_scores(radargram: Radargram, inputs: DepthInputs) -> np.ndarray | None:
    """Seam scores of one line (3-5-4).

    ``None`` (no seam rule on the line) without a zero or without zero-band
    energy (v1.4.1).
    """
    t_zero_ns = inputs.zero.t_zero_ns
    if t_zero_ns is None or radargram.dt_ns is None or radargram.dx_m is None:
        return None
    assert radargram.f_c_hz is not None
    track = zero_band_track(
        inputs.branch.filtered, t_zero_ns, radargram.dt_ns, radargram.f_c_hz
    )
    if track is None:
        return None
    lag_ns, amplitude = track
    return seam_scores(lag_ns, amplitude, radargram.f_c_hz, radargram.dx_m)


def candidate_qc(
    radargram: Radargram,
    inputs: DepthInputs,
    candidate_row: object,
    box: GridBox | None,
    result: DepthResult,
    seam_scores_line: np.ndarray | None,
) -> CandidateQc:
    """Rules B and D for one candidate (spec 3-5-2 to 3-5-4).

    Args:
        radargram: The line.
        inputs: Its ``prepare_depth_inputs`` result (the one ``result`` used).
        candidate_row: Candidate csv row with ``trace`` and ``sample``.
        box: The candidate box; ``None`` without ``tau_ns`` (curvature ``no_box``).
        result: The candidate's ``DepthResult`` (read only).
        seam_scores_line: :func:`seam_scores` of the line, ``None`` without a zero.
    """
    assert radargram.dt_ns is not None and radargram.dx_m is not None
    assert radargram.f_c_hz is not None
    trace, sample = round(float(candidate_row.trace)), float(candidate_row.sample)
    if box is None:
        curvature = CurvatureQc(QcVerdict.UNKNOWN, "no_box")
    else:
        hint = ApexHint(trace=trace, sample=sample)
        curvature = curvature_qc(radargram, box, inputs, hint, result)
    seam_score, seam_split = (
        (None, None)
        if seam_scores_line is None
        else seam_near(seam_scores_line, trace, radargram.dx_m)
    )
    return CandidateQc(
        curvature=curvature,
        record_end_margin_ns=record_end_margin_ns(
            sample, radargram.n_samples, radargram.dt_ns
        ),
        at_record_end=apex_at_record_end(
            sample, radargram.n_samples, radargram.dt_ns, radargram.f_c_hz
        ),
        seam_score=seam_score,
        seam_split=seam_split,
        on_trace_seam=seam_score is not None and seam_score >= SEAM_SCORE_LIMIT,
    )


# ----------------------------------------------------------------- depth per candidate


def _run_candidate(
    radargram: Radargram,
    inputs: DepthInputs,
    candidate: object,
    seam_scores_line: np.ndarray | None,
    apex_qc: bool,
) -> CandidateResult:
    """``estimate_depth`` for one candidate, then (optionally) its QC."""
    config = DepthConfig()
    activity_id, path = radargram.line_id.split("/", 1)
    trace, sample = round(float(candidate.trace)), float(candidate.sample)
    tau_ns = float(candidate.tau_ns)
    box = None
    if math.isnan(tau_ns):
        result = DepthResult(
            status=DepthStatus.ZERO_FAILED, flags=("candidate_tau_missing",)
        )
    else:
        box = candidate_box(radargram, trace, sample, tau_ns)
        hint = ApexHint(trace=trace, sample=sample)
        result = estimate_depth(radargram, box, config, inputs=inputs, apex_hint=hint)
    # The QC runs only after the DepthResult exists and never replaces it (3-5-6).
    qc = (
        candidate_qc(radargram, inputs, candidate, box, result, seam_scores_line)
        if apex_qc
        else None
    )
    return CandidateResult(
        activity_id=activity_id,
        path=path,
        candidate_trace=trace,
        candidate_sample=sample,
        source=str(candidate.source),
        result=result,
        candidate_moved=_candidate_moved(radargram, trace, sample, result),
        box_k0=None if box is None else box.k0,
        search_k0=None if box is None else _search_top(radargram, box, inputs.zero),
        qc=qc,
    )


def run_line(
    radargram: Radargram, candidates: pd.DataFrame, *, apex_qc: bool = True
) -> list[CandidateResult]:
    """One zero per line, ``estimate_depth`` per candidate, default settings (4-2).

    The candidate is passed as ``apex_hint`` (spec_depth_v1 5-7 item 1).

    A candidate without ``tau_ns`` (manual candidate on a line whose zero
    failed) gets no box and counts as ``zero_failed`` (spec v1.1).

    With ``apex_qc`` every candidate also gets its QC (3-5); the zero-band seam
    scores are computed once per line.
    """
    inputs = prepare_depth_inputs(radargram, DepthConfig())
    seam_scores_line = _line_seam_scores(radargram, inputs) if apex_qc else None
    return [
        _run_candidate(radargram, inputs, candidate, seam_scores_line, apex_qc)
        for candidate in candidates.itertuples(index=False)
    ]


# ----------------------------------------------------------------- ground truth


def _radius_table(metadata: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``(activity, seq) -> R`` for rows whose order matches the gt table."""
    radii = {}
    for _, row in metadata.iterrows():
        order_matches = str(row[ORDER_MATCH_COLUMN]).strip() == ORDER_MATCH_YES
        diameter = str(row["diameter_mm"]).strip()
        if order_matches and diameter:
            key = (str(row["activity_id"]), str(row["seq_meta"]))
            radii[key] = float(diameter) / 2000.0
    return radii


def build_ground_truth(
    groundtruth: pd.DataFrame,
    metadata: pd.DataFrame,
    include_code_x: bool = False,
) -> dict[str, list[GtItem]]:
    """Spec 2 on in-memory tables (all columns as text)."""
    radii = _radius_table(metadata)
    by_activity: dict[str, list[GtItem]] = defaultdict(list)
    trenches = groundtruth.groupby("activity_id")["trench_no"].nunique()
    for row in groundtruth.itertuples(index=False):
        if not str(row.depth_m).strip():
            continue
        if row.code == CODE_NOT_A_PIPE and not include_code_x:
            continue
        key = (str(row.activity_id), str(row.seq))
        radius = radii.get(key)
        by_activity[str(row.activity_id)].append(
            GtItem(
                depth_m=float(row.depth_m),
                code=str(row.code),
                radius_m=0.0 if radius is None else radius,
                radius_known=radius is not None,
                trench_no=str(row.trench_no),
                multi_trench=bool(trenches[row.activity_id] > 1),
            )
        )
    return dict(by_activity)


def build_ground_truth_sets(
    groundtruth: pd.DataFrame, metadata: pd.DataFrame
) -> dict[str, dict[str, list[GtItem]]]:
    """Both ground-truth sets of one run: ``exclude_x`` (main) and ``include_x``."""
    return {
        "exclude_x": build_ground_truth(groundtruth, metadata, include_code_x=False),
        "include_x": build_ground_truth(groundtruth, metadata, include_code_x=True),
    }


def load_ground_truth_sets(root: Path) -> dict[str, dict[str, list[GtItem]]]:
    """Read the 4TU metadata CSVs and apply spec 2 for both gt sets."""
    groundtruth = load_fourtu_table(root, FourTUTable.GROUNDTRUTH)
    metadata = load_fourtu_table(root, FourTUTable.UTILITIES_METADATA)
    return build_ground_truth_sets(groundtruth, metadata)


# ----------------------------------------------------------------- clustering


Run = tuple[int, int]
"""A cluster as a contiguous ``[start, stop)`` range of the sorted members."""


def _check_cluster_inputs(
    depths: Sequence[float], lines: Sequence[object], width_m: float, linkage: str
) -> None:
    """Input checks of spec 4-3-1 item 7 (``ValueError`` on any violation)."""
    if linkage not in CLUSTER_LINKAGES:
        raise ValueError(f"linkage must be one of {CLUSTER_LINKAGES}, got {linkage!r}")
    if len(depths) != len(lines):
        raise ValueError(f"{len(depths)} depths but {len(lines)} lines")
    if not width_m > 0:  # also rejects NaN
        raise ValueError(f"width_m must be positive, got {width_m}")
    if not all(math.isfinite(depth) for depth in depths):
        raise ValueError("depths must be finite (no NaN or inf)")


def _single_linkage_runs(sorted_depths: Sequence[float], width_m: float) -> list[Run]:
    """v1.2: split wherever two sorted neighbours are more than ``width_m`` apart."""
    runs, start = [], 0
    for index in range(1, len(sorted_depths) + 1):
        at_end = index == len(sorted_depths)
        if at_end or sorted_depths[index] - sorted_depths[index - 1] > width_m:
            runs.append((start, index))
            start = index
    return runs


def _narrowest_merge(
    sorted_depths: Sequence[float], runs: Sequence[Run], width_m: float
) -> int | None:
    """Index ``j`` of the adjacent pair ``(j, j + 1)`` to merge next, or ``None``.

    Only pairs whose merged width is at most ``width_m`` (+ edge tolerance) are
    allowed. The scan runs shallow to deep and replaces the best pair only when
    strictly narrower beyond the tie tolerance, so a tie goes to the shallower pair.
    """
    best_index, best_width_m = None, math.inf
    for index, (left, right) in enumerate(itertools.pairwise(runs)):
        merged_width_m = sorted_depths[right[1] - 1] - sorted_depths[left[0]]
        if merged_width_m > width_m + CLUSTER_EDGE_TOL_M:
            continue
        if merged_width_m < best_width_m - CLUSTER_TIE_TOL_M:
            best_index, best_width_m = index, merged_width_m
    return best_index


def _complete_linkage_runs(sorted_depths: Sequence[float], width_m: float) -> list[Run]:
    """1-D complete linkage (spec 4-3-1): merge the narrowest adjacent pair first.

    In one dimension a complete-linkage cluster is always a contiguous range of
    the sorted depths, so only adjacent runs need to be compared.
    """
    runs: list[Run] = [(index, index + 1) for index in range(len(sorted_depths))]
    while (best := _narrowest_merge(sorted_depths, runs, width_m)) is not None:
        runs[best : best + 2] = [(runs[best][0], runs[best + 1][1])]
    return runs


def cluster_depths(
    depths: Sequence[float],
    lines: Sequence[object],
    width_m: float = CLUSTER_WIDTH_M,
    linkage: str = "complete",
) -> list[Cluster]:
    """Cluster the depths of one activity across its lines (spec 4-3, 4-3-1).

    Args:
        depths: Pipe-top depths in m (finite).
        lines: Line (``path``) of each depth; same length as ``depths``.
        width_m: ``complete``: maximum cluster width ``max - min``.
            ``single``: maximum gap between sorted neighbours (v1.2).
        linkage: ``complete`` (v1.3 default) or ``single`` (v1.2, B1 test and
            diagnostics only).

    Returns:
        Clusters ordered by their shallowest member. The result does not depend
        on the input order.

    Raises:
        ValueError: Unknown ``linkage``, length mismatch, non-finite depth or
            ``width_m <= 0``.
    """
    _check_cluster_inputs(depths, lines, width_m, linkage)
    order = sorted(
        range(len(depths)), key=lambda i: (float(depths[i]), str(lines[i]), i)
    )
    sorted_depths = [float(depths[i]) for i in order]
    if linkage == "complete":
        runs = _complete_linkage_runs(sorted_depths, width_m)
    else:
        runs = _single_linkage_runs(sorted_depths, width_m)
    return [
        Cluster(
            depth_m=float(np.median(sorted_depths[start:stop])),
            n_lines=len({lines[i] for i in order[start:stop]}),
            n_apex_candidates=stop - start,
            span_m=sorted_depths[stop - 1] - sorted_depths[start],
            member_indices=tuple(order[start:stop]),
        )
        for start, stop in runs
    ]


def supported(
    clusters: Sequence[Cluster], n_lines_in_activity: int, min_lines: int
) -> list[Cluster]:
    """Support filter (4-3 item 2): activities with one line keep everything."""
    if n_lines_in_activity < 2:
        return list(clusters)
    return [cluster for cluster in clusters if cluster.n_lines >= min_lines]


# ----------------------------------------------------------------- matching, metrics


def match_activity(est: Sequence[float], gt: Sequence[float]) -> MatchResult:
    """Ungated Hungarian on ``|est - gt|`` (rectangular allowed)."""
    if len(est) == 0 or len(gt) == 0:
        return MatchResult((), missed=len(gt), false_est=len(est))
    cost = np.abs(np.subtract.outer(np.asarray(est, float), np.asarray(gt, float)))
    rows, cols = linear_sum_assignment(cost)
    pairs = tuple(
        (float(est[r]), int(c), float(est[r] - gt[c]))
        for r, c in zip(rows, cols, strict=True)
    )
    return MatchResult(
        pairs,
        missed=len(gt) - len(pairs),
        false_est=len(est) - len(pairs),
        est_indices=tuple(int(r) for r in rows),
    )


@dataclass(frozen=True)
class Summary:
    """Spec 5 metrics over a set of activities."""

    n_activities: int
    n_gt: int
    n_est: int
    n_pairs: int
    mae_m: float | None
    bias_m: float | None
    med_abs_m: float | None
    p90_abs_m: float | None
    n_pass: int
    pass_a: float | None
    pass_b: float | None
    missed: int
    false_est: int


def summarize(matches: Mapping[str, MatchResult], n_gt: Mapping[str, int]) -> Summary:
    """Pooled pair metrics; pass A per pair, pass B per ground truth (5)."""
    errors = np.array([e for m in matches.values() for _, _, e in m.pairs], dtype=float)
    total_gt = sum(n_gt[a] for a in matches)
    n_pairs = len(errors)
    n_pass = int(np.sum(np.abs(errors) <= PASS_LIMIT_M + 1e-12))
    absolute = np.abs(errors)
    return Summary(
        n_activities=len(matches),
        n_gt=total_gt,
        n_est=sum(len(m.pairs) + m.false_est for m in matches.values()),
        n_pairs=n_pairs,
        mae_m=float(absolute.mean()) if n_pairs else None,
        bias_m=float(errors.mean()) if n_pairs else None,
        med_abs_m=float(np.median(absolute)) if n_pairs else None,
        p90_abs_m=float(np.percentile(absolute, 90)) if n_pairs else None,
        n_pass=n_pass,
        pass_a=n_pass / n_pairs if n_pairs else None,
        pass_b=n_pass / total_gt if total_gt else None,
        missed=sum(m.missed for m in matches.values()),
        false_est=sum(m.false_est for m in matches.values()),
    )


# ----------------------------------------------------------------- baseline


def baseline_range(gt_depths: Sequence[float], mode: str) -> tuple[float, float]:
    """``activity``: gt range (single gt: d +- 0.3, cut at 0); ``global``: 0.2-2.0."""
    if mode == "global":
        return GLOBAL_BASELINE_RANGE_M
    low, high = min(gt_depths), max(gt_depths)
    if len(gt_depths) == 1:
        low, high = low - SINGLE_GT_HALF_RANGE_M, high + SINGLE_GT_HALF_RANGE_M
    return max(low, 0.0), max(high, 0.0)


@dataclass(frozen=True)
class BaselineDraws:
    """Per-trial, per-activity counts of one baseline run (spec 6).

    Keeping the counts per activity lets every scope (overall, per site) be
    aggregated from the **same** trials instead of drawing again.
    """

    activities: tuple[str, ...]
    n_gt: np.ndarray
    """``(activities,)`` ground-truth count."""
    n_pass: np.ndarray
    """``(trials, activities)`` pairs with ``|e| <= 0.30 m``."""
    n_pairs: np.ndarray
    """``(trials, activities)`` matched pairs."""
    abs_error_sum: np.ndarray
    """``(trials, activities)`` sum of ``|e|`` over the pairs."""


def draw_baseline(
    n_est_by_act: Mapping[str, int],
    gt_by_act: Mapping[str, Sequence[float]],
    trials: int = BASELINE_TRIALS,
    seed: int = BASELINE_SEED,
    mode: str = "activity",
) -> BaselineDraws:
    """Same number of uniform random depths per activity, same matching (6).

    Draw order is trial-major, activities in ``gt_by_act`` order, from one
    ``default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    activities = tuple(a for a in gt_by_act if a in n_est_by_act)
    shape = (trials, len(activities))
    n_pass = np.zeros(shape, dtype=int)
    n_pairs = np.zeros(shape, dtype=int)
    abs_error_sum = np.zeros(shape, dtype=float)
    for trial in range(trials):
        for column, activity in enumerate(activities):
            low, high = baseline_range(gt_by_act[activity], mode)
            draws = rng.uniform(low, high, size=n_est_by_act[activity])
            match = match_activity(list(draws), list(gt_by_act[activity]))
            errors = np.abs([error for _, _, error in match.pairs])
            n_pass[trial, column] = int(np.sum(errors <= PASS_LIMIT_M + 1e-12))
            n_pairs[trial, column] = len(errors)
            abs_error_sum[trial, column] = float(errors.sum())
    return BaselineDraws(
        activities=activities,
        n_gt=np.array([len(gt_by_act[a]) for a in activities], dtype=int),
        n_pass=n_pass,
        n_pairs=n_pairs,
        abs_error_sum=abs_error_sum,
    )


def _ratio_per_trial(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio, NaN where the denominator is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def baseline_stats(
    draws: BaselineDraws, activities: Sequence[str] | None = None
) -> BaselineStats | None:
    """Pooled baseline metrics over ``activities`` (default: all) per trial.

    Returns ``None`` when none of ``activities`` is in the draws.
    """
    wanted = set(draws.activities if activities is None else activities)
    columns = [i for i, a in enumerate(draws.activities) if a in wanted]
    if not columns:
        return None
    n_pass = draws.n_pass[:, columns].sum(axis=1)
    n_pairs = draws.n_pairs[:, columns].sum(axis=1)
    total_gt = int(draws.n_gt[columns].sum())
    maes = _ratio_per_trial(draws.abs_error_sum[:, columns].sum(axis=1), n_pairs)
    pass_a = _ratio_per_trial(n_pass, n_pairs)
    pass_b = _ratio_per_trial(n_pass, np.full(n_pass.shape, total_gt))
    return BaselineStats(
        mae_mean=float(np.nanmean(maes)),
        mae_p025=float(np.nanpercentile(maes, 2.5)),
        mae_p975=float(np.nanpercentile(maes, 97.5)),
        pass_a_mean=float(np.nanmean(pass_a)),
        pass_b_mean=float(np.nanmean(pass_b)),
        pass_b_p025=float(np.nanpercentile(pass_b, 2.5)),
        pass_b_p975=float(np.nanpercentile(pass_b, 97.5)),
        pass_a_p025=float(np.nanpercentile(pass_a, 2.5)),
        pass_a_p975=float(np.nanpercentile(pass_a, 97.5)),
        pass_b_values=pass_b,
    )


def random_baseline(
    n_est_by_act: Mapping[str, int],
    gt_by_act: Mapping[str, Sequence[float]],
    trials: int = BASELINE_TRIALS,
    seed: int = BASELINE_SEED,
    mode: str = "activity",
) -> BaselineStats | None:
    """Baseline statistics over all activities of one draw (spec 6)."""
    return baseline_stats(draw_baseline(n_est_by_act, gt_by_act, trials, seed, mode))


# ----------------------------------------------------------------- estimates


def estimates_by_activity(
    candidates: Sequence[CandidateResult],
    zero_variant: str,
    variant: str,
    lines_per_activity: Mapping[str, int],
    width_m: float,
    linkage: str,
    min_lines: int,
    apex_qc: bool = True,
) -> tuple[dict[str, list[float]], list[dict[str, object]]]:
    """Per-activity estimates for one zero variant and variant (4-3, 3-5-5).

    With ``apex_qc``, every variant but ``noqc`` drops the QC-excluded
    candidates **before** clustering, so an excluded candidate never lends its
    line to another candidate's support: ``main``, ``a`` and ``b`` use the
    B+D+E rule, ``bd`` the B+D rule (v1.5). Candidates without a ``qc``
    attribute (or ``qc`` ``None``) are kept. ``bd`` and ``noqc`` then cluster
    and filter like ``main``.
    """
    applies_qc = apex_qc and variant != NO_QC_VARIANT
    depths: dict[str, list[float]] = defaultdict(list)
    lines: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if applies_qc and _is_qc_excluded(candidate, variant):
            continue
        depth = candidate.depth(zero_variant)
        if depth is not None:
            depths[candidate.activity_id].append(depth)
            lines[candidate.activity_id].append(candidate.path)
    estimates: dict[str, list[float]] = {}
    cluster_rows: list[dict[str, object]] = []
    for activity in sorted(lines_per_activity):
        if variant == "b":
            estimates[activity] = list(depths.get(activity, []))
            continue
        clusters = cluster_depths(
            depths.get(activity, []), lines.get(activity, []), width_m, linkage
        )
        kept = supported(clusters, lines_per_activity[activity], min_lines)
        uses_support = variant in ("main", BD_VARIANT, NO_QC_VARIANT)
        estimates[activity] = [c.depth_m for c in (kept if uses_support else clusters)]
        if variant == "main":
            for index, cluster in enumerate(clusters):
                cluster_rows.append(
                    {
                        "activity_id": activity,
                        "site": site_of_activity(activity),
                        "zero_variant": zero_variant,
                        "cluster_id": index,
                        "depth_m": round(cluster.depth_m, 4),
                        "n_lines": cluster.n_lines,
                        "n_apex_candidates": cluster.n_apex_candidates,
                        "kept_by_support": cluster in kept,
                        "span_m": round(cluster.span_m, CLUSTER_SPAN_DECIMALS),
                        "linkage": linkage,
                        "width_m": width_m,
                    }
                )
    return estimates, cluster_rows


def _is_qc_excluded(candidate: object, variant: str) -> bool:
    """Whether ``variant``'s QC rule drops ``candidate`` (3-5-5 item 1, v1.5).

    ``main``, ``a`` and ``b`` use B+D+E (``excluded``), ``bd`` uses B+D
    (``excluded_bd``) and ``noqc`` drops nothing. A candidate without QC is kept.
    """
    qc = getattr(candidate, "qc", None)
    if qc is None or variant == NO_QC_VARIANT:
        return False
    if variant == BD_VARIANT:
        return qc.excluded_bd
    return qc.excluded


def apex_qc_enabled(args: argparse.Namespace) -> bool:
    """``--apex-qc on`` (the default)."""
    return args.apex_qc == "on"


def variants_for(args: argparse.Namespace) -> tuple[str, ...]:
    """Reported variants: ``bd`` and ``noqc`` only exist when the QC runs
    (3-5-5 item 7)."""
    if apex_qc_enabled(args):
        return VARIANTS
    return tuple(v for v in VARIANTS if v not in QC_AUX_VARIANTS)


@dataclass(frozen=True)
class LinkageComparison:
    """Estimate counts of one run under complete and single linkage (spec 9-1).

    Both counts come from the same depths (detected zero, ``main`` variant)
    after the support filter; the single counts are diagnostics only and never
    enter csv, baselines or the verdict.
    """

    complete_counts: dict[str, int]
    single_counts: dict[str, int]
    gt_activities: frozenset[str]
    """Activities with ground truth in the main set (``exclude_x``)."""


def compare_with_single_linkage(
    candidates: Sequence[CandidateResult],
    lines_per_activity: Mapping[str, int],
    complete_estimates: Mapping[str, list[float]],
    gt_activities: frozenset[str],
    args: argparse.Namespace,
) -> LinkageComparison | None:
    """Re-cluster the main combination with single linkage (spec 9-1, v1.3.1).

    Returns:
        ``None`` when the run itself uses single linkage, otherwise the counts
        of ``complete_estimates`` next to the single-linkage counts obtained
        with the same ``width_m`` and ``min_lines``.
    """
    if args.cluster_linkage != "complete":
        return None
    single_estimates, _ = estimates_by_activity(
        candidates,
        "detected",
        "main",
        lines_per_activity,
        width_m=args.cluster_width_m,
        linkage="single",
        min_lines=args.min_support_lines,
        apex_qc=apex_qc_enabled(args),
    )
    return LinkageComparison(
        complete_counts={a: len(v) for a, v in complete_estimates.items()},
        single_counts={a: len(v) for a, v in single_estimates.items()},
        gt_activities=gt_activities,
    )


# ----------------------------------------------------------------- evaluation


@dataclass
class Evaluation:
    """Everything written to the outputs."""

    candidate_rows: list[dict[str, object]] = field(default_factory=list)
    cluster_rows: list[dict[str, object]] = field(default_factory=list)
    match_rows: list[dict[str, object]] = field(default_factory=list)
    summary_rows: list[dict[str, object]] = field(default_factory=list)
    no_gt_counts: dict[str, int] = field(default_factory=dict)
    site11_rows: list[dict[str, object]] = field(default_factory=list)
    site11_summaries: dict[tuple[str, str, str], Summary] = field(default_factory=dict)
    multi_trench_activities: list[str] = field(default_factory=list)
    linkage_comparison: LinkageComparison | None = None
    estimate_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    """Detected-zero estimate count per activity, keyed by ``main`` / ``bd`` /
    ``noqc`` (v1.5; ``bd`` and ``noqc`` only with ``--apex-qc on``)."""
    gt_activities: frozenset[str] = frozenset()
    """Activities with ground truth in the main set (``exclude_x``)."""


@dataclass(frozen=True)
class Combination:
    """One reported combination (spec 5)."""

    gt_set: str
    assumption: str
    zero_variant: str
    variant: str


def _summary_row(
    scope: str,
    combination: Combination,
    summary: Summary,
    baselines: Mapping[str, BaselineStats | None],
) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "gt_set": combination.gt_set,
        "assumption": combination.assumption,
        "zero_variant": combination.zero_variant,
        "variant": combination.variant,
        **{
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in summary.__dict__.items()
            if key != "n_pass"
        },
    }
    row.update(_baseline_columns("baseline", baselines.get("activity"), summary))
    row.update(_baseline_columns("baseline_global", baselines.get("global"), summary))
    return row


def _baseline_columns(
    prefix: str, stats: BaselineStats | None, summary: Summary
) -> dict[str, object]:
    """Baseline csv columns (spec 8, v1.2).

    The activity-range baseline reports MAE, pass A and pass B; the global
    range only pass B and ``p``. Blank when there is no baseline.
    """
    values: dict[str, float | None] = {}
    if prefix == "baseline":
        values |= {
            "mae_mean": None if stats is None else stats.mae_mean,
            "mae_p025": None if stats is None else stats.mae_p025,
            "mae_p975": None if stats is None else stats.mae_p975,
            "pass_a_mean": None if stats is None else stats.pass_a_mean,
            "pass_a_p025": None if stats is None else stats.pass_a_p025,
            "pass_a_p975": None if stats is None else stats.pass_a_p975,
        }
    values |= {
        "pass_b_mean": None if stats is None else stats.pass_b_mean,
        "pass_b_p025": None if stats is None else stats.pass_b_p025,
        "pass_b_p975": None if stats is None else stats.pass_b_p975,
        "p": None
        if stats is None or summary.pass_b is None
        else float(np.mean(stats.pass_b_values >= summary.pass_b)),
    }
    return {
        f"{prefix}_{name}": "" if value is None else round(value, 4)
        for name, value in values.items()
    }


def baseline_trials_for(variant: str, args: argparse.Namespace) -> dict[str, int]:
    """Trials per baseline mode (spec 6, v1.1).

    Main variant: activity range ``--baseline-trials``, global range
    ``--baseline-trials-aux``. Auxiliary variants (a, b, bd, noqc): activity range
    ``--baseline-trials-aux`` only.
    """
    if variant == "main":
        return {"activity": args.baseline_trials, "global": args.baseline_trials_aux}
    return {"activity": args.baseline_trials_aux}


def evaluate_combinations(
    candidates: Sequence[CandidateResult],
    gt_sets: Mapping[str, Mapping[str, list[GtItem]]],
    lines_per_activity: Mapping[str, int],
    args: argparse.Namespace,
) -> Evaluation:
    """All gt set x assumption x zero x variant combinations, overall and per site.

    Depths and clusters do not depend on the gt set; only matching does.
    """
    evaluation = Evaluation(gt_activities=frozenset(gt_sets["exclude_x"]))
    for zero_variant in ZERO_VARIANTS:
        for variant in variants_for(args):
            estimates, cluster_rows = estimates_by_activity(
                candidates,
                zero_variant,
                variant,
                lines_per_activity,
                width_m=args.cluster_width_m,
                linkage=args.cluster_linkage,
                min_lines=args.min_support_lines,
                apex_qc=apex_qc_enabled(args),
            )
            evaluation.cluster_rows.extend(cluster_rows)
            if zero_variant == "detected" and variant in COUNTED_VARIANTS:
                evaluation.estimate_counts[variant] = {
                    a: len(v) for a, v in estimates.items()
                }
            if zero_variant == "detected" and variant == "main":
                main_gt = gt_sets["exclude_x"]
                evaluation.no_gt_counts = {
                    a: len(v) for a, v in estimates.items() if a not in main_gt
                }
                evaluation.linkage_comparison = compare_with_single_linkage(
                    candidates,
                    lines_per_activity,
                    estimates,
                    frozenset(main_gt),
                    args,
                )
            for gt_set, ground_truth in gt_sets.items():
                for assumption in ASSUMPTIONS:
                    combination = Combination(gt_set, assumption, zero_variant, variant)
                    _evaluate_one(
                        evaluation, estimates, ground_truth, combination, args
                    )
    return evaluation


def _match_rows(
    matches: Mapping[str, MatchResult],
    ground_truth: Mapping[str, list[GtItem]],
    gt_depths: Mapping[str, list[float]],
    combination: Combination,
) -> list[dict[str, object]]:
    rows = []
    for activity, match in matches.items():
        for est, gt_index, error in match.pairs:
            item = ground_truth[activity][gt_index]
            rows.append(
                {
                    "activity_id": activity,
                    "site": site_of_activity(activity),
                    "gt_set": combination.gt_set,
                    "multi_trench": item.multi_trench,
                    "assumption": combination.assumption,
                    "zero_variant": combination.zero_variant,
                    "variant": combination.variant,
                    "est_depth_m": round(est, 4),
                    "gt_depth_m": round(gt_depths[activity][gt_index], 4),
                    "gt_code": item.code,
                    "radius_known": item.radius_known,
                    "error_m": round(error, 4),
                    "within_30cm": abs(error) <= PASS_LIMIT_M + 1e-12,
                }
            )
    return rows


def _evaluate_one(
    evaluation: Evaluation,
    estimates: Mapping[str, list[float]],
    ground_truth: Mapping[str, list[GtItem]],
    combination: Combination,
    args: argparse.Namespace,
) -> None:
    gt_depths = {
        a: [item.depth_for(combination.assumption) for item in items]
        for a, items in ground_truth.items()
        if a in estimates
    }
    matches = {a: match_activity(estimates[a], gt_depths[a]) for a in gt_depths}
    evaluation.match_rows.extend(
        _match_rows(matches, ground_truth, gt_depths, combination)
    )
    n_gt = {a: len(v) for a, v in gt_depths.items()}
    n_est = {a: len(estimates[a]) for a in matches}
    # One draw per baseline mode; every scope is aggregated from these trials.
    draws = {
        mode: draw_baseline(n_est, gt_depths, trials, args.baseline_seed, mode)
        for mode, trials in baseline_trials_for(combination.variant, args).items()
    }
    sites = sorted({site_of_activity(a) for a in matches})
    scopes = [("all", list(matches))] + [
        (f"site_{s}", [a for a in matches if site_of_activity(a) == s]) for s in sites
    ]
    for scope, activities in scopes:
        subset = {a: matches[a] for a in activities}
        baselines = {
            mode: baseline_stats(draw, activities) for mode, draw in draws.items()
        }
        evaluation.summary_rows.append(
            _summary_row(scope, combination, summarize(subset, n_gt), baselines)
        )


@dataclass(frozen=True)
class Site11Estimate:
    """One site 11 diagnostic estimate (spec 7)."""

    depth_eps8_m: float
    """Depth rescaled to eps_r 8 (clustered on this value)."""
    depth_m: float
    """Unscaled detected-zero depth (median of the same members for a cluster)."""


def _depth_at_eps8(candidate: CandidateResult) -> float | None:
    """Detected-zero depth rescaled to eps_r 8: ``d * sqrt(eps_used / 8)``."""
    depth, eps = candidate.depth("detected"), candidate.result.eps_r_used
    if depth is None or eps is None:
        return None
    return depth * math.sqrt(eps / SITE_11_DIAGNOSTIC_EPS)


def site11_estimates(
    candidates: Sequence[CandidateResult],
    lines_per_activity: Mapping[str, int],
    variant: str,
    width_m: float,
    linkage: str,
    min_lines: int,
    apex_qc: bool = True,
) -> dict[str, list[Site11Estimate]]:
    """Site 11 estimates per activity: rescale first, then the 4-3 path.

    ``main_path`` clusters the rescaled depths and applies the support filter
    exactly like the main result; ``unclustered`` keeps every candidate.
    With ``apex_qc`` both drop the candidates excluded by the main rule
    (B+D+E, v1.5) before rescaling (3-5-5 item 8), so they see the candidate
    sets of ``main`` and ``b``. The QC verdicts are the run's (metadata eps_r),
    never re-judged at eps_r 8.
    """
    scaled: dict[str, list[float]] = defaultdict(list)
    unscaled: dict[str, list[float]] = defaultdict(list)
    paths: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if site_of_activity(candidate.activity_id) != SITE_LONG_DT:
            continue
        if apex_qc and _is_qc_excluded(candidate, "main"):
            continue
        depth_eps8 = _depth_at_eps8(candidate)
        depth = candidate.depth("detected")
        if depth_eps8 is None or depth is None:
            continue
        scaled[candidate.activity_id].append(depth_eps8)
        unscaled[candidate.activity_id].append(depth)
        paths[candidate.activity_id].append(candidate.path)
    estimates: dict[str, list[Site11Estimate]] = {}
    for activity in sorted(lines_per_activity):
        if site_of_activity(activity) != SITE_LONG_DT:
            continue
        activity_scaled, activity_unscaled = scaled[activity], unscaled[activity]
        if variant == "unclustered":
            estimates[activity] = [
                Site11Estimate(depth_eps8, depth)
                for depth_eps8, depth in zip(
                    activity_scaled, activity_unscaled, strict=True
                )
            ]
            continue
        clusters = cluster_depths(activity_scaled, paths[activity], width_m, linkage)
        kept = supported(clusters, lines_per_activity[activity], min_lines)
        estimates[activity] = [
            Site11Estimate(
                cluster.depth_m,
                float(
                    np.median([activity_unscaled[i] for i in cluster.member_indices])
                ),
            )
            for cluster in kept
        ]
    return estimates


def site11_diagnostic(
    candidates: Sequence[CandidateResult],
    gt_sets: Mapping[str, Mapping[str, list[GtItem]]],
    lines_per_activity: Mapping[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], Summary]]:
    """Site 11 eps_r 8 diagnostic (spec 7, v1.1/v1.2); never used for judging.

    Returns:
        ``depth_4tu_v1_site11.csv`` rows and summaries keyed by
        ``(gt_set, assumption, variant)``. Kept apart from the main metrics.
    """
    rows: list[dict[str, object]] = []
    summaries: dict[tuple[str, str, str], Summary] = {}
    for variant in SITE11_VARIANTS:
        estimates = site11_estimates(
            candidates,
            lines_per_activity,
            variant,
            width_m=args.cluster_width_m,
            linkage=args.cluster_linkage,
            min_lines=args.min_support_lines,
            apex_qc=apex_qc_enabled(args),
        )
        for gt_set, ground_truth in gt_sets.items():
            for assumption in ASSUMPTIONS:
                gt_depths = {
                    a: [item.depth_for(assumption) for item in ground_truth[a]]
                    for a in estimates
                    if a in ground_truth
                }
                matches = {
                    a: match_activity(
                        [e.depth_eps8_m for e in estimates[a]], gt_depths[a]
                    )
                    for a in gt_depths
                }
                summaries[(gt_set, assumption, variant)] = summarize(
                    matches, {a: len(v) for a, v in gt_depths.items()}
                )
                for activity, match in matches.items():
                    for (_, gt_index, error), est_index in zip(
                        match.pairs, match.est_indices, strict=True
                    ):
                        estimate = estimates[activity][est_index]
                        rows.append(
                            {
                                "activity_id": activity,
                                "gt_set": gt_set,
                                "assumption": assumption,
                                "zero_variant": "detected",
                                "variant": variant,
                                "est_depth_m": round(estimate.depth_m, 4),
                                "est_depth_eps8_m": round(estimate.depth_eps8_m, 4),
                                "gt_depth_m": round(gt_depths[activity][gt_index], 4),
                                "error_eps8_m": round(error, 4),
                                "within_30cm_eps8": abs(error) <= PASS_LIMIT_M + 1e-12,
                            }
                        )
    return rows, summaries


# ----------------------------------------------------------------- outputs


def candidate_row(
    candidate: CandidateResult, lines: Mapping[tuple[str, str], Mapping[str, object]]
) -> dict[str, object]:
    result = candidate.result
    est = result.alt_depths.get(META_ZERO_KEYS["est"])
    act = result.alt_depths.get(META_ZERO_KEYS["act"])
    qc = result.eps_r_refraction_qc_range
    line = lines.get((candidate.activity_id, candidate.path), {})
    flags = list(result.flags)
    if float(line.get("length_m", 0) or 0) >= LONG_LINE_M:
        flags.append("long_line")
    if candidate.qc is not None:
        flags.extend(candidate.qc.flags)  # only here, never in DepthResult.flags
    return {
        "activity_id": candidate.activity_id,
        "site": site_of_activity(candidate.activity_id),
        "path": candidate.path,
        "candidate_trace": candidate.candidate_trace,
        "candidate_sample": round(candidate.candidate_sample, 2),
        "source": candidate.source,
        "status": result.status.value,
        "depth_top_m": ""
        if result.depth_top_m is None
        else round(result.depth_top_m, 4),
        "depth_est_m": "" if est is None else round(est.depth_top_m, 4),
        "depth_act_m": "" if act is None else round(act.depth_top_m, 4),
        "label_est": "" if est is None else est.label,
        "label_act": "" if act is None else act.label,
        "eps_r_used": result.eps_r_used if result.eps_r_used is not None else "",
        "zero_rule": result.zero_rule or "",
        "t_zero_ns": "" if result.t_zero_ns is None else round(result.t_zero_ns, 4),
        "h_est_m": "" if result.h_est_m is None else round(result.h_est_m, 4),
        "apex_trace": "" if result.apex_trace is None else result.apex_trace,
        "apex_sample": ""
        if result.apex_sample is None
        else round(result.apex_sample, 2),
        "candidate_moved": candidate.candidate_moved,
        "eps_qc_lo": "" if qc is None else round(qc[0], 3),
        "eps_qc_hi": "" if qc is None else round(qc[1], 3),
        "flags": " ".join(flags),
        **_box_top_columns(candidate),
        **_qc_columns(candidate.qc),
        **_tracking_columns(candidate),
    }


def _rounded(value: float | None, digits: int) -> object:
    """Blank for a missing value, ``inf`` spelled out, else rounded."""
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return round(value, digits)


def _qc_columns(qc: CandidateQc | None) -> dict[str, object]:
    """Candidates csv QC columns (spec 8); all blank with ``--apex-qc off``."""
    if qc is None:
        return dict.fromkeys(QC_COLUMNS, "")
    curvature = qc.curvature
    return {
        "qc_verdict": curvature.verdict.value,
        "qc_reason": curvature.reason,
        "qc_r": _rounded(curvature.r, 4),
        "qc_v_curv_m_per_ns": _rounded(curvature.v_curv_m_per_ns, 4),
        "qc_sigma_v_rel": _rounded(curvature.sigma_v_rel, 4),
        "qc_inlier_frac": _rounded(curvature.inlier_frac, 4),
        "qc_n_picks": curvature.n_picks,
        "qc_n_left": curvature.n_left,
        "qc_n_right": curvature.n_right,
        "qc_aperture_m": _rounded(curvature.aperture_m, 3),
        "record_end_margin_ns": _rounded(qc.record_end_margin_ns, 3),
        "apex_at_record_end": qc.at_record_end,
        "seam_score": _rounded(qc.seam_score, 3),
        "seam_split": "" if qc.seam_split is None else qc.seam_split,
        "apex_on_trace_seam": qc.on_trace_seam,
        "qc_excluded": qc.excluded,
    }


def _tracking_columns(candidate: CandidateResult) -> dict[str, object]:
    """Tracking diagnostic columns (``TRACKING_COLUMNS``, spec_depth v1.9 13-4)."""
    is_descended = APEX_DESCENDED in candidate.result.flags
    if candidate.qc is None:
        return {"stop_left": "", "stop_right": "", "apex_descended": is_descended}
    curvature = candidate.qc.curvature
    return {
        "stop_left": _stop_text(curvature.stop_left),
        "stop_right": _stop_text(curvature.stop_right),
        "apex_descended": is_descended,
    }


def _stop_text(stop: TrackStop | None) -> str:
    return "" if stop is None else stop.value


def _box_top_columns(candidate: CandidateResult) -> dict[str, object]:
    """Diagnostic box-top columns (spec_depth_v1 5-7 item 5): blank without a box."""
    apex_sample = candidate.result.apex_sample
    apex_minus_search_k0 = (
        ""
        if candidate.search_k0 is None or apex_sample is None
        else round(apex_sample - candidate.search_k0, 2)
    )
    return {
        "box_k0": "" if candidate.box_k0 is None else candidate.box_k0,
        "search_k0": "" if candidate.search_k0 is None else candidate.search_k0,
        "apex_minus_search_k0": apex_minus_search_k0,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return lines


def _clustering_setting(args: argparse.Namespace) -> str:
    """md run-condition phrase for the clustering (spec 9-1)."""
    if args.cluster_linkage == "complete":
        meaning = "묶음 폭 상한(완전 연결)"
    else:
        meaning = "이웃 간격 상한(단일 연결, v1.2 진단용)"
    return (
        f"묶기 `linkage={args.cluster_linkage}`, `width_m={args.cluster_width_m}` "
        f"({meaning} {args.cluster_width_m} m)"
    )


def _count_change_phrase(
    label: str, comparison: LinkageComparison, activities: Sequence[str]
) -> str:
    """One md sub-line: count sums and how many activities gained or lost."""
    complete_total = sum(comparison.complete_counts[a] for a in activities)
    single_total = sum(comparison.single_counts[a] for a in activities)
    differences = [
        comparison.complete_counts[a] - comparison.single_counts[a] for a in activities
    ]
    n_more = sum(difference > 0 for difference in differences)
    n_same = sum(difference == 0 for difference in differences)
    n_fewer = sum(difference < 0 for difference in differences)
    return (
        f"  - {label} {len(activities)}개: complete {complete_total} / "
        f"single {single_total} / 차 {complete_total - single_total:+d} "
        f"(활동 수: 늘어남 {n_more}, 같음 {n_same}, 줄어듦 {n_fewer})"
    )


def _linkage_comparison_lines(comparison: LinkageComparison | None) -> list[str]:
    """md line "v1.2 대비 추정 수" (spec 4-3-2, 9-1, v1.3.1)."""
    if comparison is None:
        return ["- v1.2 대비 추정 수: 해당 없음(실행 자체가 single)"]
    all_activities = sorted(comparison.complete_counts)
    gt_activities = [a for a in all_activities if a in comparison.gt_activities]
    return [
        "- v1.2 대비 추정 수(같은 깊이를 `linkage=single`, 같은 `width_m`·"
        "지지 측선으로 다시 묶어 지지 필터 뒤 셈, 판정에 쓰지 않음):",
        _count_change_phrase("전체 활동", comparison, all_activities),
        _count_change_phrase("정답 활동(`exclude_x`)", comparison, gt_activities),
    ]


def _cluster_diagnostic_lines(
    cluster_rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    linkage_comparison: LinkageComparison | None,
) -> list[str]:
    """md cluster diagnostics of the main combination (detected zero, spec 9-1).

    Reported only; ``width_m`` is never re-chosen from these values (4-3-2).
    """
    rows = [row for row in cluster_rows if row["zero_variant"] == "detected"]
    header = ["## 묶음 진단 (주 결과 조합: 검출 영점, 판정에 쓰지 않음)", ""]
    comparison_lines = _linkage_comparison_lines(linkage_comparison)
    if not rows:
        return [*header, "- 묶음 없음", *comparison_lines, ""]
    spans_m = np.array([float(row["span_m"]) for row in rows])
    span_quantiles = np.percentile(spans_m, SPAN_QUANTILES_PERCENT).round(4).tolist()
    over_width = int(np.sum(spans_m > args.cluster_width_m + CLUSTER_EDGE_TOL_M))
    n_lines_counts = Counter(int(row["n_lines"]) for row in rows)
    n_kept = sum(bool(row["kept_by_support"]) for row in rows)
    percent_labels = "/".join(str(p) for p in SPAN_QUANTILES_PERCENT)
    return [
        *header,
        f"- 묶음 {len(rows)}개 (`linkage={args.cluster_linkage}`, "
        f"`width_m={args.cluster_width_m}`)",
        f"- `span_m` 분위({percent_labels} %): {span_quantiles}",
        f"- `span_m` > `width_m` + {CLUSTER_EDGE_TOL_M:g} m: {over_width}개"
        + (" (완전 연결이면 0이어야 함)" if args.cluster_linkage == "complete" else ""),
        "- `n_lines` 분포: "
        + ", ".join(f"{n}: {count}" for n, count in sorted(n_lines_counts.items())),
        f"- 지지 필터 통과: {n_kept}/{len(rows)} ({n_kept / len(rows):.3f})",
        *comparison_lines,
        "",
    ]


def _qc_setting(args: argparse.Namespace) -> str:
    """md run-condition line of the apex-candidate QC (spec 8 item 1)."""
    if not apex_qc_enabled(args):
        return (
            "- 꼭짓점 후보 QC: `apex_qc=off`(주 결과 = v1.3.1 경로, `bd`·`noqc` 없음)"
        )
    return (
        f"- 꼭짓점 후보 QC: `apex_qc=on`, 곡률 띠 {QC_R_LO}-{QC_R_HI}"
        f"(R {QC_RADIUS_M} m, unstable: 정상점 < {QC_UNSTABLE_INLIER} 또는 "
        f"sigma_v/v_m > {QC_UNSTABLE_SIGMA_REL}), "
        f"기록 끝 T_end {RECORD_END_PERIODS} 주기, "
        f"이음 theta_t {SEAM_TIME_PERIODS} 주기·theta_A ln {SEAM_AMP_RATIO}·"
        f"k {SEAM_NOISE_K:g}(이음 창 ±{SEAM_APEX_HALF_M} m). "
        "묶기 전에 주 결과(B+D+E)는 곡률 out·unknown·기록 끝·이음 후보를, "
        "보조 `bd`(B+D)는 out·기록 끝·이음 후보를 뺌"
    )


def _auxiliary_qc_table_lines(
    bd_rows: Sequence[Mapping[str, object]],
    noqc_rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    args: argparse.Namespace,
) -> list[str]:
    """``bd`` then ``noqc`` rows right under the main table, same format
    (spec 5 and 8, v1.5)."""
    if not apex_qc_enabled(args):
        return []
    return [
        "### 보조: B+D (`bd` = 곡률 unknown 유지, v1.4.4까지의 주 결과 경로, "
        "지지 필터, 전체, X 제외)",
        "",
        *_md_table(bd_rows, columns),
        "",
        "### 보조: QC 없음 (`noqc` = v1.3.1 주 결과 경로, 지지 필터, 전체, X 제외)",
        "",
        *_md_table(noqc_rows, columns),
        "",
    ]


def _verdict_phrase(qcs: Sequence[CandidateQc]) -> str:
    counts = Counter(qc.curvature.verdict.value for qc in qcs)
    return " / ".join(f"{v.value} {counts.get(v.value, 0)}" for v in QcVerdict)


def _reason_phrase(qcs: Sequence[CandidateQc]) -> str:
    counts = Counter(
        (qc.curvature.verdict.value, qc.curvature.reason)
        for qc in qcs
        if qc.curvature.verdict != QcVerdict.IN
    )
    if not counts:
        return "없음"
    return ", ".join(
        f"{verdict} `{reason}` {n}" for (verdict, reason), n in sorted(counts.items())
    )


def _grouped_verdict_lines(
    label: str, keyed_qcs: Sequence[tuple[object, CandidateQc]]
) -> list[str]:
    """One md sub-line per group (source or site): verdict counts."""
    groups: dict[object, list[CandidateQc]] = defaultdict(list)
    for key, qc in keyed_qcs:
        groups[key].append(qc)
    return [
        f"  - {label} {key}: {_verdict_phrase(qcs)}"
        for key, qcs in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def _ratio_quantile_phrase(qcs: Sequence[CandidateQc]) -> str:
    """Quantiles of ``r`` over the ``in`` and ``out`` verdicts (spec 8, v1.4.1).

    An ``unknown`` verdict may still carry an ``r`` (``unstable`` fits), but
    that ``r`` is not trusted, so it stays out of the quantiles.
    """
    ratios = [
        qc.curvature.r
        for qc in qcs
        if qc.curvature.verdict != QcVerdict.UNKNOWN and qc.curvature.r is not None
    ]
    if not ratios:
        return "알려진 r 없음"
    percent_labels = "/".join(str(p) for p in QC_R_QUANTILES_PERCENT)
    quantiles = np.percentile(ratios, QC_R_QUANTILES_PERCENT).round(4).tolist()
    return f"알려진 r {len(ratios)}개, 분위({percent_labels} %): {quantiles}"


def _rule_d_phrase(qcs: Sequence[CandidateQc]) -> str:
    """Rule D counts and their overlap with a curvature ``out``."""
    record_end = [qc for qc in qcs if qc.at_record_end]
    seam = [qc for qc in qcs if qc.on_trace_seam]
    both = sum(qc.at_record_end and qc.on_trace_seam for qc in qcs)

    def with_out(selected: Sequence[CandidateQc]) -> int:
        return sum(qc.curvature.verdict == QcVerdict.OUT for qc in selected)

    return (
        f"`{FLAG_RECORD_END}` {len(record_end)}(B out과 겹침 {with_out(record_end)}), "
        f"`{FLAG_SEAM}` {len(seam)}(B out과 겹침 {with_out(seam)}), 둘 다 {both}"
    )


def _stop_count_phrase(stops: Sequence[TrackStop | None]) -> str:
    counts = Counter("none" if stop is None else stop.value for stop in stops)
    return ", ".join(f"`{name}` {n}" for name, n in sorted(counts.items())) or "없음"


def _tracking_phrase(candidates: Sequence[CandidateResult]) -> str:
    """Apex descent rate and wing stop reasons of the ``ok`` candidates (v1.9)."""
    curvatures = [c.qc.curvature for c in candidates if c.qc is not None]
    descended = sum(APEX_DESCENDED in c.result.flags for c in candidates)
    left = _stop_count_phrase([q.stop_left for q in curvatures])
    right = _stop_count_phrase([q.stop_right for q in curvatures])
    return (
        f"`{APEX_DESCENDED}` {descended}/{len(candidates)}; "
        f"정지 사유 왼쪽 {left} / 오른쪽 {right}"
    )


def _estimate_count_phrase(
    label: str, evaluation: Evaluation, activities: Sequence[str]
) -> str:
    """Estimate totals of ``main``, ``bd`` and ``noqc`` and the differences
    from ``main`` (spec 8 v1.5 (v))."""

    def total(variant: str) -> int:
        counts = evaluation.estimate_counts[variant]
        return sum(counts.get(a, 0) for a in activities)

    main_total = total("main")
    bd_total = total(BD_VARIANT)
    noqc_total = total(NO_QC_VARIANT)
    return (
        f"  - {label} {len(activities)}개: main {main_total} / bd {bd_total} "
        f"/ noqc {noqc_total} / main-bd {main_total - bd_total:+d} "
        f"/ main-noqc {main_total - noqc_total:+d}"
    )


def _exclusion_lines(ok: Sequence[CandidateResult]) -> list[str]:
    """``qc_excluded`` counts under B+D+E and B+D, and the extra E drops by
    source (spec 8 v1.5 (iv)); ``ok`` candidates all carry a QC."""
    qcs = [c.qc for c in ok if c.qc is not None]
    excluded_main = sum(qc.excluded for qc in qcs)
    excluded_bd = sum(qc.excluded_bd for qc in qcs)
    extra_by_source = Counter(
        c.source
        for c in ok
        if c.qc is not None and c.qc.excluded and not c.qc.excluded_bd
    )
    by_source = (
        ", ".join(f"`{source}` {n}" for source, n in sorted(extra_by_source.items()))
        or "없음"
    )
    return [
        f"- `qc_excluded`(주 결과 B+D+E): {excluded_main}/{len(qcs)}; "
        f"B+D(`bd`): {excluded_bd}/{len(qcs)}; "
        f"E로 더 뺀 후보(곡률 unknown, D 없음): {excluded_main - excluded_bd}",
        f"  - E로 더 뺀 후보 출처별: {by_source}",
    ]


def _qc_diagnostic_lines(
    candidates: Sequence[CandidateResult],
    evaluation: Evaluation,
    args: argparse.Namespace,
) -> list[str]:
    """md section 8 (v1.5): QC verdicts of the ``ok`` candidates, never re-tuned."""
    header = ["## 꼭짓점 후보 QC 진단 (v1.5, 판정에 쓰지 않음)", ""]
    if not apex_qc_enabled(args):
        return [*header, "- `apex_qc=off`: QC를 계산하지 않았다.", ""]
    ok = [c for c in candidates if c.result.status == "ok" and c.qc is not None]
    qcs = [c.qc for c in ok if c.qc is not None]
    activities = sorted(evaluation.estimate_counts.get("main", {}))
    gt_activities = [a for a in activities if a in evaluation.gt_activities]
    return [
        *header,
        f"- `ok` 후보 {len(qcs)}개의 곡률 판정: {_verdict_phrase(qcs)}",
        f"- 사유: {_reason_phrase(qcs)}",
        "- 출처별:",
        *_grouped_verdict_lines("source", [(c.source, c.qc) for c in ok]),
        "- 현장별:",
        *_grouped_verdict_lines(
            "site", [(site_of_activity(c.activity_id), c.qc) for c in ok]
        ),
        f"- {_ratio_quantile_phrase(qcs)}",
        f"- D 플래그: {_rule_d_phrase(qcs)}",
        *_exclusion_lines(ok),
        f"- 추적 진단(v1.9, 판정 미사용): {_tracking_phrase(ok)}",
        "- 추정 수(검출 영점, 지지 필터 뒤):",
        _estimate_count_phrase("전체 활동", evaluation, activities),
        _estimate_count_phrase("정답 활동(`exclude_x`)", evaluation, gt_activities),
        f"- 사전 등록 문구(spec §11 항목 3): {PREREGISTERED_QC_NOTE}",
        "",
    ]


def write_markdown(
    path: Path,
    evaluation: Evaluation,
    candidates: Sequence[CandidateResult],
    args: argparse.Namespace,
    elapsed_s: float,
) -> None:
    from gpr_detection.run_info import git_state, local_timestamp

    git = git_state(PROJECT_ROOT)
    main_variant = [r for r in evaluation.summary_rows if r["variant"] == "main"]
    main_rows = [
        r for r in main_variant if r["scope"] == "all" and r["gt_set"] == "exclude_x"
    ]

    def overall_rows(variant: str) -> list[dict[str, object]]:
        return [
            r
            for r in evaluation.summary_rows
            if r["variant"] == variant
            and r["scope"] == "all"
            and r["gt_set"] == "exclude_x"
        ]

    bd_rows = overall_rows(BD_VARIANT)
    noqc_rows = overall_rows(NO_QC_VARIANT)
    with_x_rows = [
        r for r in main_variant if r["scope"] == "all" and r["gt_set"] == "include_x"
    ]
    site_rows = [
        r
        for r in main_variant
        if r["scope"] != "all"
        and r["gt_set"] == "exclude_x"
        and r["zero_variant"] == "detected"
    ]
    site11_summary_rows = [
        {
            "assumption": assumption,
            "variant": variant,
            **{
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in summary.__dict__.items()
            },
        }
        for (gt_set, assumption, variant), summary in sorted(
            evaluation.site11_summaries.items()
        )
        if gt_set == "exclude_x"
    ]
    statuses = Counter(s.result.status.value for s in candidates)
    moved = Counter((s.source, s.candidate_moved) for s in candidates)
    site9 = [
        s for s in candidates if site_of_activity(s.activity_id) == SITE_TWO_EVENTS
    ]
    h_est = [s.result.h_est_m for s in site9 if s.result.h_est_m is not None]
    suspects = sum("meta_eps_suspect" in s.result.flags for s in site9)
    h_quantiles = (
        np.percentile(h_est, [0, 25, 50, 75, 100]).round(3).tolist() if h_est else "-"
    )
    columns = (
        "assumption",
        "zero_variant",
        "n_gt",
        "n_est",
        "n_pairs",
        "mae_m",
        "bias_m",
        "pass_a",
        "pass_b",
        "missed",
        "false_est",
        "baseline_pass_b_mean",
        "baseline_pass_b_p025",
        "baseline_pass_b_p975",
        "baseline_p",
    )
    lines = [
        "# 4TU 깊이 검증 v1 (M2-8, spec_eval_4tu_v1)",
        "",
        f"- 실행: {local_timestamp()}, commit {git['commit']} (dirty {git['dirty']}), "
        f"{elapsed_s:.0f} s, 단일 프로세스",
        "- 설정: `DepthConfig()` 기본값(4TU 튜닝 없음), "
        f"{_clustering_setting(args)}, 지지 측선 >= {args.min_support_lines}, "
        f"`default_rng({args.baseline_seed})`, 기준선 시행: 주 결과 조합 활동 범위 "
        f"{args.baseline_trials}회, 보조 조합({', '.join(variants_for(args)[1:])})과 "
        f"전역 범위 {args.baseline_trials_aux}회(현장별 값은 같은 시행에서 집계)",
        _qc_setting(args),
        "- 정답 집합: 주 결과 `exclude_x`(code X 제외), 보조 `include_x` "
        "(같은 실행의 매칭 단계에서만 다름, summary·matches의 `gt_set` 열)",
        f"- 꼭짓점 후보 {len(candidates)}개, "
        f"정답 없는 활동 {len(evaluation.no_gt_counts)}개"
        f"(주 결과 추정 {sum(evaluation.no_gt_counts.values())}개, 오검출로 세지 않음)",
        "",
        "## 주 결과 (지지 필터, 전체, X 제외)",
        "",
        *_md_table(main_rows, columns),
        "",
        *_auxiliary_qc_table_lines(bd_rows, noqc_rows, columns, args),
        "## 보조: X 포함 (지지 필터, 전체)",
        "",
        *_md_table(with_x_rows, columns),
        "",
        "## 현장별 (주 결과, 검출 영점, X 제외)",
        "",
        *_md_table(site_rows, ("scope", "assumption", *columns[2:])),
        "",
        *_cluster_diagnostic_lines(
            evaluation.cluster_rows, args, evaluation.linkage_comparison
        ),
        "## 상태·candidate_moved·출처",
        "",
        *[f"- status `{k}`: {v}" for k, v in statuses.most_common()],
        *[
            f"- source {src}, candidate_moved {mv}: {n}"
            for (src, mv), n in sorted(moved.items())
        ],
        "",
        "## 현장 9 (두 사건 측선)",
        "",
        f"- h_est 분위: {h_quantiles}",
        f"- meta_eps_suspect: {suspects}/{len(site9)}",
        "",
        "## 현장 11 진단 (eps_r 8 환산, 판정에 쓰지 않음)",
        "",
        "꼭짓점 후보 깊이를 먼저 환산한 뒤 주 결과와 같은 묶기·지지 필터·매칭"
        "(`main_path`, 검출 영점, X 제외). `unclustered`는 묶지 않은 보조 결과. "
        f"짝별 값은 `{args.prefix}_site11.csv`.",
        "",
        *_md_table(
            site11_summary_rows,
            (
                "assumption",
                "variant",
                "n_gt",
                "n_est",
                "n_pairs",
                "mae_m",
                "bias_m",
                "pass_b",
            ),
        ),
        "",
        "## 가정·한계",
        "",
        "- 보조 결과(a: 지지 필터 없음, b: 묶지 않음"
        + (
            ", bd: B+D(곡률 unknown 유지), noqc: QC 없음"
            if apex_qc_enabled(args)
            else ""
        )
        + ")와 t0 est/act는 csv에 있다.",
        "- 꼭짓점 후보를 `apex_hint`로 넘긴다(spec_depth_v1 §5-7). 힌트 창"
        "(±3 트레이스·±0.5 주기)이 `candidate_moved` 기준과 같으므로 "
        "`candidate_moved`는 거의 항상 False다. 위 끝 경계 효과는 "
        "candidates csv의 `apex_minus_search_k0`와 플래그 "
        "`apex_at_box_top`·`apex_hint_unmatched`로 본다.",
        "- `box_in_zero_zone`이면 박스가 뮤트 아래로 거의 남지 않아 탐색을 "
        "하지 않으므로 `search_k0`는 실제 탐색 위 끝이 아닐 수 있다.",
        "- 두 트렌치 활동(정답 합집합, 같은 관이 두 번 들어갈 수 있음, matches의 "
        "`multi_trench`): " + (", ".join(evaluation.multi_trench_activities) or "없음"),
        "- 무작위 기준선 주 = 활동별 정답 범위, 보조 = 전역 0.2-2.0 m.",
        "- 메타 eps_r은 실측 보증이 없다. 현장 1·4·7·12·13은 헤더 기본값과 같다.",
        "",
        *_qc_diagnostic_lines(candidates, evaluation, args),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- entry point


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="4TU package root")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_APEX_CANDIDATES)
    parser.add_argument("--lines", type=Path, default=DEFAULT_LINES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--activities", default=None, help="comma list, e.g. 01.1,04.1")
    parser.add_argument(
        "--cluster-width-m",
        type=float,
        default=CLUSTER_WIDTH_M,
        help="complete: maximum cluster width; single: maximum neighbour gap",
    )
    parser.add_argument(
        "--cluster-linkage",
        choices=CLUSTER_LINKAGES,
        default="complete",
        help="single = v1.2 chaining, diagnostics only",
    )
    parser.add_argument("--min-support-lines", type=int, default=MIN_SUPPORT_LINES)
    parser.add_argument(
        "--baseline-trials",
        type=int,
        default=BASELINE_TRIALS,
        help="activity-range baseline trials of the main variant",
    )
    parser.add_argument(
        "--baseline-trials-aux",
        type=int,
        default=BASELINE_TRIALS_AUX,
        help="trials of variants a/b and of the global-range baseline",
    )
    parser.add_argument("--baseline-seed", type=int, default=BASELINE_SEED)
    parser.add_argument(
        "--apex-qc",
        choices=("on", "off"),
        default="on",
        help="apex-candidate QC before clustering (v1.4); off = v1.3.1 main result",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse candidates, lines and ground truth only; load no radargram",
    )
    return parser.parse_args(argv)


def _read_inputs(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = args.root or load_dataset_paths().fourtu
    candidates = pd.read_csv(args.candidates, dtype={"activity_id": str, "path": str})
    lines = pd.read_csv(args.lines, dtype={"activity_id": str, "path": str})
    if args.activities:
        wanted = set(args.activities.split(","))
        candidates = candidates[candidates["activity_id"].isin(wanted)]
        lines = lines[lines["activity_id"].isin(wanted)]
    return root, candidates, lines


def dry_run(args: argparse.Namespace) -> int:
    root, candidates, lines = _read_inputs(args)
    gt_sets = load_ground_truth_sets(root)
    ground_truth = gt_sets["exclude_x"]
    activities = sorted(lines["activity_id"].unique())
    report = {
        "candidates": len(candidates),
        "candidates_by_source": candidates["source"].value_counts().to_dict(),
        "candidates_without_tau": int(candidates["tau_ns"].isna().sum()),
        "activities": len(activities),
        "lines": len(lines),
        "lines_with_candidates": int(
            candidates.groupby(["activity_id", "path"]).ngroups
        ),
        "activities_with_gt": sum(a in ground_truth for a in activities),
        "activities_without_gt": sum(a not in ground_truth for a in activities),
        "gt_items": sum(len(ground_truth.get(a, [])) for a in activities),
        "gt_items_include_x": sum(
            len(gt_sets["include_x"].get(a, [])) for a in activities
        ),
        "gt_radius_known": sum(
            i.radius_known for a in activities for i in ground_truth.get(a, [])
        ),
        "multi_trench_activities": sorted(
            a
            for a in activities
            if any(i.multi_trench for i in ground_truth.get(a, []))
        ),
        "long_lines": int((lines["length_m"] >= LONG_LINE_M).sum()),
    }
    print(json.dumps(report, ensure_ascii=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    started = time.perf_counter()
    root, candidates_table, lines_table = _read_inputs(args)
    gt_sets = load_ground_truth_sets(root)
    lines_per_activity = lines_table.groupby("activity_id")["path"].nunique().to_dict()
    line_info = {
        (r.activity_id, r.path): r._asdict()
        for r in lines_table.itertuples(index=False)
    }

    candidates: list[CandidateResult] = []
    for (activity, path), group in candidates_table.groupby(
        ["activity_id", "path"], sort=True
    ):
        radargram = load_fourtu_radargram(root, activity, path, T0Choice.BOTH)
        candidates.extend(run_line(radargram, group, apex_qc=apex_qc_enabled(args)))
        LOGGER.info("%s/%s: %d candidates", activity, path, len(group))

    evaluation = evaluate_combinations(candidates, gt_sets, lines_per_activity, args)
    evaluation.candidate_rows = [candidate_row(s, line_info) for s in candidates]
    evaluation.site11_rows, evaluation.site11_summaries = site11_diagnostic(
        candidates, gt_sets, lines_per_activity, args
    )
    evaluation.multi_trench_activities = sorted(
        a
        for a, items in gt_sets["exclude_x"].items()
        if a in lines_per_activity and any(i.multi_trench for i in items)
    )
    elapsed_s = time.perf_counter() - started

    out, prefix = args.out_dir, args.prefix
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{prefix}_candidates.csv", evaluation.candidate_rows)
    write_csv(out / f"{prefix}_clusters.csv", evaluation.cluster_rows)
    write_csv(out / f"{prefix}_matches.csv", evaluation.match_rows)
    write_csv(out / f"{prefix}_site11.csv", evaluation.site11_rows)
    write_csv(out / f"{prefix}_summary.csv", evaluation.summary_rows)
    write_markdown(out / f"{prefix}.md", evaluation, candidates, args, elapsed_s)
    LOGGER.info("done in %.0f s", elapsed_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
