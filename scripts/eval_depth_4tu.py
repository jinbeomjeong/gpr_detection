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
* gate (v1.6, auxiliary only): +-0.30 m gated matching (max pairs, then min
  sum |e|) gives recall, precision and F1 (spec 5-1); never used for the verdict.
* duplicate radargrams (v1.7): within-activity copies dropped before
  evaluation; across-activity duplicates kept in the main result, sensitivity
  keep_one/drop_all in a separate csv (never used for the verdict). The
  committed list ``docs/datasets/fourtu_duplicate_radargrams.csv`` is checked
  against the file hashes before any depth is computed (spec 4-5).
* ``zero_h_implausible`` (v1.7): a line flag only (``air_surface_late`` zero
  with ``h_est > h_max``); depths, exclusions and metrics ignore it (spec 7-1).
  Since spec_depth v2.1 4TU has no ``air_surface_late`` line, so 0 is normal.
* two-event zero lines (v1.8): lines whose zero rule is ``air_first_event``
  are counted in an md subsection of the site 9 section; report only.

* tie range and unobservable truths (v1.9, auxiliary only): pass-count range
  over co-optimal assignments (spec 12-1) and truths below the record end on
  every evaluated line (12-2); 11 columns at the end of the summary rows, two
  new csvs and one md section. The K1 draft (12-3) was removed after the S1g
  verdict (docs/processing/m210_k1_s1g_judgment.md); reproduce it at commit
  90845b4.

Outputs: ``<prefix>_candidates.csv``, ``_clusters.csv``, ``_matches.csv``,
``_site11.csv``, ``_summary.csv``, ``_dup_sensitivity.csv`` (v1.7),
``_ties.csv``, ``_unobservable.csv`` (v1.9) and ``<prefix>.md``.

Truth-dependent code stays here, not in ``src``. ``--dry-run`` only parses the
candidate, line, ground-truth and duplicate tables (no radargram is loaded).

Example:
    conda run -n gpr-dev python -I scripts/eval_depth_4tu.py --dry-run
    conda run -n gpr-dev python -I scripts/eval_depth_4tu.py
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

# MKL without conditional numerical reproducibility changes the last bits of the
# depths between processes, and the exact-tie matchings of the auxiliary variants
# then pick other pairs (M2-10, 2026-11-11). It must be set before numpy loads
# MKL; an explicit value from the caller wins.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.depth.estimate import (
    DepthInputs,
    depth_from_time,
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
from gpr_detection.preprocess.timezero import (
    AIR_MULTI_EVENT,
    DEFAULT_PERIOD_NS,
    ZeroResult,
    ZeroRule,
)

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
PASS_TOL_M = 1e-12
"""Float slack of the pass predicate (0.45 - 0.15 = 0.30000000000000004 passes).
Shared by the pass rates (spec 5) and the gate (5-1) through ``_within_pass_limit``."""
GATE_PENALTY_OFFSET_M = 1.0
"""The "+1" of the gate penalty P = m * g + 1 (spec 5-1): finite, yet larger than
any assignment cost inside the gate."""
TIE_COST_TOL_M = 1e-9
"""Co-optimal band on the sum of |e| (spec 12-1-2): assignments within 1e-9 m of
the optimum tie (float noise of a 19-truth activity is ~1e-14 m)."""

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
GATE_SUMMARY_FIELDS = (
    "gate_pairs",
    "gate_recall",
    "gate_precision",
    "gate_f1",
    "gate_mae_m",
)
"""Observed gate metrics of ``Summary`` (spec 5-1, v1.6; auxiliary only)."""
GATE_COLUMNS = (
    *GATE_SUMMARY_FIELDS,
    "baseline_gate_recall_mean",
    "baseline_gate_recall_p025",
    "baseline_gate_recall_p975",
    "baseline_gate_precision_mean",
    "baseline_gate_precision_p025",
    "baseline_gate_precision_p975",
    "baseline_gate_f1_mean",
    "baseline_gate_f1_p025",
    "baseline_gate_f1_p975",
    "baseline_gate_p",
    "baseline_gate_pct",
    "baseline_global_gate_f1_mean",
    "baseline_global_gate_f1_p025",
    "baseline_global_gate_f1_p975",
    "baseline_global_gate_p",
)
"""The 20 gate columns at the end of every summary csv row (spec 8, v1.6)."""
GATE_MAIN_TABLE_COLUMNS = (
    "assumption",
    "zero_variant",
    "n_gt",
    "n_est",
    "pass_b",
    *GATE_SUMMARY_FIELDS,
    "baseline_gate_f1_mean",
    "baseline_gate_f1_p025",
    "baseline_gate_f1_p975",
    "baseline_gate_p",
    "baseline_gate_pct",
)
"""md gate section table (ii): ``main``, overall, X excluded (spec 8, v1.6)."""
GATE_VARIANT_TABLE_COLUMNS = (
    "variant",
    "assumption",
    "n_est",
    "gate_pairs",
    "gate_recall",
    "gate_precision",
    "gate_f1",
    "baseline_gate_f1_mean",
    "baseline_gate_p",
)
"""md gate section table (iii): every variant, detected zero (spec 8, v1.6)."""

# Zero flag and duplicate radargrams (spec 7-1 and 4-5, v1.7). The list path is
# not a CLI option: the pre-registered list cannot be swapped or switched off.
FLAG_ZERO_H = "zero_h_implausible"
DEFAULT_DUPLICATES = (
    PROJECT_ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
)
DUP_WITHIN = "within_activity"
DUP_ACROSS = "across_activities"
DUP_SCOPES = (DUP_WITHIN, DUP_ACROSS)
DUPLICATE_TABLE_COLUMNS = (
    "pair_id",
    "sha256_npy",
    "activity_a",
    "path_a",
    "activity_b",
    "path_b",
    "scope",
)
"""Columns the duplicate list must have; other columns are ignored (4-5-2)."""
DUPLICATE_CSV_COLUMNS = ("duplicate_radargram", "duplicate_of")
"""Last two candidates csv columns (spec 8, v1.7)."""
KEEP_BOTH, KEEP_ONE, DROP_ALL = "keep_both", "keep_one", "drop_all"
DUP_HANDLINGS = (KEEP_BOTH, KEEP_ONE, DROP_ALL)
"""Across-activity duplicate handling; ``keep_both`` is the main result (4-5-4)."""
KEEP_REASON_SUPERSET = "strict_superset"
KEEP_REASON_EQUAL = "equal_sets"
KEEP_REASON_LOWER = "lower_number"
DUP_CAVEAT_PASS_B_PP = 2.0
DUP_CAVEAT_TOL_PP = 1e-9
"""Caveat when any |delta pass B| >= 2 %p (+-1e-9; user decision a, spec 4-5-5)."""
HASH_BLOCK_BYTES = 1 << 20
RADARGRAM_DIR = "05_radargrams_npy"
DUP_SENSITIVITY_TABLE_COLUMNS = (
    "dup_handling",
    "assumption",
    "n_activities",
    "n_gt",
    "n_est",
    "mae_m",
    "pass_a",
    "pass_b",
    "d_pass_b_pp",
    "d_mae_m",
    "baseline_pass_b_mean",
    "baseline_p",
)
"""md sensitivity table (spec 8 v1.7 (ii)); ``d_*`` against ``keep_both``."""
ZERO_FLAG_TABLE_COLUMNS = (
    "site",
    "lines",
    "surface_late",
    "flagged",
    "candidates",
    "ok",
    "main_input",
    "flagged_gt_lines",
)
"""md zero-flag table in the site 9 section (spec 8 v1.7 (iii))."""
MULTI_EVENT_TABLE_COLUMNS = (
    "site",
    "lines",
    "multi_event",
    "candidates",
    "ok",
    "main_input",
    "multi_event_gt_lines",
)
"""md two-event zero table after the zero-flag table (spec 7-1 / 8 (iii) v1.8)."""

# Tie range and unobservable truths (spec 12-1 and 12-2, v1.9; auxiliary only).
# K4 has no constant of its own: it reuses rule D (RECORD_END_*, 12-2-1).
OBS_UNOBSERVABLE = "unobservable"
OBS_UNKNOWN = "window_unknown"
AUX_COLUMNS = (
    "n_pass",
    "tie_pass_min",
    "tie_pass_max",
    "tie_pass_b_min",
    "tie_pass_b_max",
    "tie_activities",
    "n_gt_unobservable",
    "n_pass_unobservable",
    "n_gt_window_unknown",
    "n_gt_observable",
    "pass_b_observable",
)
"""The 11 v1.9 columns after ``baseline_global_gate_p`` (spec 9-8 C)."""
TIE_CSV_COLUMNS = (
    "variant",
    "zero_variant",
    "gt_set",
    "assumption",
    "activity_id",
    "site",
    "n_est",
    "n_gt",
    "n_pass",
    "tie_pass_min",
    "tie_pass_max",
)
"""``<prefix>_ties.csv`` (spec 12-1-5)."""
UNOBSERVABLE_CSV_COLUMNS = (
    "zero_variant",
    "gt_set",
    "assumption",
    "activity_id",
    "site",
    "gt_index",
    "gt_depth_m",
    "gt_code",
    "observability",
    "depth_limit_m",
    "n_windows",
    "pass_main",
)
"""``<prefix>_unobservable.csv`` (spec 12-2-4)."""
TIE_TABLE_COLUMNS = (
    "assumption",
    "zero_variant",
    "n_gt",
    "n_pass",
    "tie_pass_min",
    "tie_pass_max",
    "pass_b",
    "tie_pass_b_min",
    "tie_pass_b_max",
    "tie_activities",
)
"""md tie-range table (spec 12-1-5 item 2)."""
UNOBSERVABLE_TABLE_COLUMNS = (
    "assumption",
    "zero_variant",
    "n_gt",
    "n_gt_unobservable",
    "n_pass_unobservable",
    "n_gt_window_unknown",
    "pass_b",
    "pass_b_observable",
)
"""md unobservable-truth table (spec 12-1-5 item 4)."""

PREREG_NEAR_APEX_STEPS = 4
"""Pre-registered value (spec 11 item 6 / 9-8 G), not a verdict threshold: a wing
"stopped within 4 traces of the apex" stopped on one of tracking steps 1-4."""
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
class RecordWindow:
    """Record window of one line for the unobservable-truth rule (spec 12-2-1).

    A property of the line, not of a candidate: every candidate of a line holds
    the same window.
    """

    n_samples: int
    dt_ns: float
    f_c_hz: float
    v_m_per_ns: float | None
    """``velocity_from_eps(radargram.eps_r_meta)``; ``None`` without a metadata
    eps_r."""
    air_gap: str | None
    """``radargram.air_gap`` (4TU ``"air"``)."""
    offset_m: float | None
    """``radargram.antenna_offset_m`` (4TU ``None``)."""
    t_zero_ns: Mapping[str, float | None]
    """Zero time per ``ZERO_VARIANTS`` key: ``detected`` = ``inputs.zero.t_zero_ns``,
    ``est`` / ``act`` = the metadata zero ``inputs.zero.meta_t0[...][0]``."""


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
    zero_h_implausible: bool = False
    """Line flag of 7-1 (same value for every candidate of a line); report only."""
    record_window: RecordWindow | None = None
    """Line record window (12-2, v1.9); only the unobservable-truth report reads it."""

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
    gate_pairs_values: np.ndarray = field(repr=False)
    """Per-trial gate pair count K_b over the scope (v1.6)."""
    n_est_total: int
    """Estimates over the scope; fixed across trials (same count as observed)."""
    n_gt_total: int


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

    The ``zero_h_implausible`` line flag (7-1, v1.7) is computed once from the
    line's zero and copied onto every candidate; the depth calls do not see it.
    The line's ``record_window`` (12-2, v1.9) is attached the same way.
    """
    config = DepthConfig()
    inputs = prepare_depth_inputs(radargram, config)
    is_zero_h_implausible = zero_h_implausible(
        inputs.zero.zero_rule, inputs.zero.h_est_m, config.h_max_m
    )
    seam_scores_line = _line_seam_scores(radargram, inputs) if apex_qc else None
    results = [
        _run_candidate(radargram, inputs, candidate, seam_scores_line, apex_qc)
        for candidate in candidates.itertuples(index=False)
    ]
    window = record_window(radargram, inputs)
    return [
        replace(result, zero_h_implausible=is_zero_h_implausible, record_window=window)
        for result in results
    ]


def zero_h_implausible(
    zero_rule: str | None,
    h_est_m: float | None,
    h_max_m: float = DEFAULT_DEPTH_CONFIG.h_max_m,
) -> bool:
    """Zero flag of spec 7-1: ``air_surface_late`` zero with ``h_est > h_max``.

    The limit is the zero search's own maximum antenna height (no new
    constant); ``h_est == h_max`` is still allowed by the search, so it is not
    flagged. Pure; ``zero_rule`` may be a ``ZeroRule`` or its string value.
    """
    if zero_rule != ZeroRule.AIR_SURFACE_LATE or h_est_m is None:
        return False
    return h_est_m > h_max_m


# ----------------------------------------------------------------- record window (12-2)


def record_window(radargram: Radargram, inputs: DepthInputs) -> RecordWindow:
    """The line's record window for the unobservable-truth rule (spec 12-2-1).

    Raises:
        ValueError: The line has no ``dt_ns`` or ``f_c_hz``.
    """
    if radargram.dt_ns is None or radargram.f_c_hz is None:
        raise ValueError(f"{radargram.line_id}: record window needs dt_ns and f_c_hz")
    zero = inputs.zero
    t_zero_ns: dict[str, float | None] = {"detected": zero.t_zero_ns}
    for zero_variant, key in META_ZERO_KEYS.items():
        meta = zero.meta_t0.get(key)
        t_zero_ns[zero_variant] = None if meta is None else meta[0]
    eps_r = radargram.eps_r_meta
    return RecordWindow(
        n_samples=radargram.n_samples,
        dt_ns=radargram.dt_ns,
        f_c_hz=radargram.f_c_hz,
        v_m_per_ns=None if eps_r is None else velocity_from_eps(eps_r),
        air_gap=radargram.air_gap,
        offset_m=radargram.antenna_offset_m,
        t_zero_ns=t_zero_ns,
    )


def time_from_depth(
    depth_m: float, v_m_per_ns: float, *, air_gap: str | None, offset_m: float | None
) -> float:
    """Two-way time after the zero for a pipe-top depth (inverse of
    ``depth_from_time``, spec 12-2-1).

    Raises:
        ValueError: ``depth_m < 0`` or ``v_m_per_ns <= 0``.
    """
    if depth_m < 0:
        raise ValueError(f"depth must be >= 0, got {depth_m}")
    if not v_m_per_ns > 0:
        raise ValueError(f"velocity must be positive, got {v_m_per_ns}")
    if air_gap == "air" or offset_m is None:
        return 2.0 * depth_m / v_m_per_ns
    return 2.0 * math.hypot(depth_m, offset_m / 2.0) / v_m_per_ns


def _usable_zero(window: RecordWindow, zero_variant: str) -> float | None:
    """The window's zero time, or ``None`` if the window cannot be used."""
    if window.v_m_per_ns is None:
        return None
    return window.t_zero_ns.get(zero_variant)


def _below_record_end(depth_m: float, window: RecordWindow, t_zero_ns: float) -> bool:
    """Rule D (3-5-3) on the sample where a truth at ``depth_m`` would appear."""
    assert window.v_m_per_ns is not None
    tau_ns = time_from_depth(
        depth_m, window.v_m_per_ns, air_gap=window.air_gap, offset_m=window.offset_m
    )
    sample = (t_zero_ns + tau_ns) / window.dt_ns
    return apex_at_record_end(sample, window.n_samples, window.dt_ns, window.f_c_hz)


def truth_observability(
    depth_m: float, windows: Sequence[RecordWindow], zero_variant: str
) -> str | None:
    """Spec 12-2-1: ``None`` (observable), ``OBS_UNOBSERVABLE`` or ``OBS_UNKNOWN``.

    Unobservable when every usable window puts the truth in the record-end zone
    of rule D; unknown when no window is usable (no metadata velocity or no zero
    of ``zero_variant``).

    A negative truth depth (centre assumption with a radius larger than the
    depth, 4TU 08.4 / 08.5) is judged as ``max(d, 0)``; the truth value itself
    is not changed (spec 12-2-1).
    """
    effective_depth_m = max(depth_m, 0.0)
    verdicts = []
    for window in windows:
        t_zero_ns = _usable_zero(window, zero_variant)
        if t_zero_ns is not None:
            verdicts.append(_below_record_end(effective_depth_m, window, t_zero_ns))
    if not verdicts:
        return OBS_UNKNOWN
    return OBS_UNOBSERVABLE if all(verdicts) else None


def depth_limit_m(window: RecordWindow, zero_variant: str) -> float | None:
    """Deepest depth outside the record-end zone (report only, spec 9-8 B).

    ``None`` when the window is unusable or the limit is negative.
    """
    t_zero_ns = _usable_zero(window, zero_variant)
    if t_zero_ns is None:
        return None
    assert window.v_m_per_ns is not None
    end_zone_ns = RECORD_END_PERIODS * 1e9 / window.f_c_hz
    time_ns = (window.n_samples - 1) * window.dt_ns - end_zone_ns - t_zero_ns
    if time_ns < 0:
        return None
    return depth_from_time(
        window.v_m_per_ns, time_ns, air_gap=window.air_gap, offset_m=window.offset_m
    )


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


def _within_pass_limit(abs_error: float | np.ndarray) -> bool | np.ndarray:
    """``|e| <= 0.30 m`` with the float slack (spec 5 pass rates, 5-1 gate)."""
    return abs_error <= PASS_LIMIT_M + PASS_TOL_M


def match_activity(est: Sequence[float], gt: Sequence[float]) -> MatchResult:
    """Ungated Hungarian on ``|est - gt|``, rectangular (spec 5; the verdict)."""
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
class GateMatch:
    """+-0.30 m gated matching of one activity (spec 5-1, v1.6; auxiliary only)."""

    n_est: int
    n_gt: int
    pairs: tuple[tuple[int, int, float], ...] = ()
    """``(est index, gt index, error = est - gt)`` inside the gate, est-index order.
    One optimum: under exact ties the pair identity is not unique, the count
    and the |e| sum are."""

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def abs_error_sum(self) -> float:
        return float(sum(abs(error) for _, _, error in self.pairs))


def match_activity_gated(est: Sequence[float], gt: Sequence[float]) -> GateMatch:
    """Max pairs with |e| <= 0.30 m, then min sum |e| (spec 5-1).

    Linear assignment on cost |e| inside the gate and P = m * g + 1 outside;
    only pairs inside the gate are kept (exactness proof in spec 5-1).
    """
    if len(est) == 0 or len(gt) == 0:
        return GateMatch(len(est), len(gt))
    errors = np.subtract.outer(np.asarray(est, float), np.asarray(gt, float))
    abs_errors = np.abs(errors)
    inside_gate = _within_pass_limit(abs_errors)
    # Finite on purpose: inf makes the assignment infeasible when a row is all
    # outside the gate, and a huge constant blurs the |e| tie-break (spec 5-1).
    penalty_m = min(abs_errors.shape) * PASS_LIMIT_M + GATE_PENALTY_OFFSET_M
    rows, cols = linear_sum_assignment(np.where(inside_gate, abs_errors, penalty_m))
    pairs = tuple(
        (int(r), int(c), float(errors[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if inside_gate[r, c]
    )
    return GateMatch(len(est), len(gt), pairs)


@dataclass(frozen=True)
class TieRange:
    """Pass count of one activity over its co-optimal assignments (spec 12-1)."""

    n_pass: int
    """Official count: scipy's default solution, predicate ``_within_pass_limit``."""
    pass_min: int
    """Minimum over the co-optimal assignments (12-1-1)."""
    pass_max: int


def _n_pass(match: MatchResult) -> int:
    return sum(bool(_within_pass_limit(abs(error))) for _, _, error in match.pairs)


def _perturbed_assignment(
    cost: np.ndarray, passes: np.ndarray, lam_m: float
) -> tuple[int, float]:
    """Pass count and sum |e| of ``argmin [S(sigma) + lam k(sigma)]`` (12-1-2)."""
    rows, cols = linear_sum_assignment(cost + lam_m * passes)
    return int(passes[rows, cols].sum()), float(cost[rows, cols].sum())


def tie_range(
    est: Sequence[float], gt: Sequence[float], match: MatchResult | None = None
) -> TieRange:
    """Pass-count range over the co-optimal ungated assignments (spec 12-1-2).

    Two linear assignments on ``cost -/+ lam P`` with ``lam = TIE_COST_TOL_M /
    (m + 1)`` find the most and the fewest passing pairs among the assignments
    whose sum |e| ties with the optimum. The official assignment is not re-solved.

    Args:
        est: Estimated depths (m).
        gt: Ground-truth depths (m).
        match: The official ``MatchResult`` of ``(est, gt)``; computed with
            ``match_activity`` when ``None``.

    Raises:
        RuntimeError: A perturbed solution leaves the tie band or the official
            count lies outside ``[pass_min, pass_max]`` (both code defects).
    """
    if len(est) == 0 or len(gt) == 0:
        return TieRange(0, 0, 0)
    if match is None:
        match = match_activity(est, gt)
    cost = np.abs(np.subtract.outer(np.asarray(est, float), np.asarray(gt, float)))
    passes = np.asarray(_within_pass_limit(cost), dtype=float)
    lam_m = TIE_COST_TOL_M / (min(cost.shape) + 1)
    n_pass = _n_pass(match)
    official_sum_m = float(sum(abs(error) for _, _, error in match.pairs))
    pass_max, sum_max_m = _perturbed_assignment(cost, passes, -lam_m)
    pass_min, sum_min_m = _perturbed_assignment(cost, passes, lam_m)
    if max(sum_max_m, sum_min_m) > official_sum_m + TIE_COST_TOL_M:
        raise RuntimeError(
            f"tie solution sum {max(sum_max_m, sum_min_m)} m exceeds the official "
            f"{official_sum_m} m + {TIE_COST_TOL_M} m (spec 12-1-2)"
        )
    if not pass_min <= n_pass <= pass_max:
        raise RuntimeError(
            f"official pass count {n_pass} outside [{pass_min}, {pass_max}] "
            "(spec 12-1-2)"
        )
    return TieRange(n_pass=n_pass, pass_min=pass_min, pass_max=pass_max)


@dataclass(frozen=True)
class Summary:
    """Spec 5 metrics over a set of activities (+ spec 5-1 gate metrics, v1.6)."""

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
    gate_pairs: int | None = None
    gate_recall: float | None = None
    gate_precision: float | None = None
    gate_f1: float | None = None
    gate_mae_m: float | None = None
    tie_pass_min: int | None = None
    tie_pass_max: int | None = None
    tie_activities: int | None = None
    n_gt_unobservable: int | None = None
    n_pass_unobservable: int | None = None
    n_gt_window_unknown: int | None = None


AUX_SUMMARY_FIELDS = (
    "tie_pass_min",
    "tie_pass_max",
    "tie_activities",
    "n_gt_unobservable",
    "n_pass_unobservable",
    "n_gt_window_unknown",
)
"""v1.9 ``Summary`` fields (12-1, 12-2); written only by ``_aux_columns``."""


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _check_gate_matches(
    matches: Mapping[str, MatchResult],
    n_gt: Mapping[str, int],
    gate_matches: Mapping[str, GateMatch],
) -> None:
    """Same activities and counts as the ungated matches; property 1 (spec 5-1).

    Raises:
        ValueError: activity keys, estimate or ground-truth counts differ.
        RuntimeError: fewer gate pairs than ungated passing pairs (a code defect).
    """
    if set(gate_matches) != set(matches):
        raise ValueError("gate_matches must cover exactly the activities of matches")
    for activity, match in matches.items():
        gate = gate_matches[activity]
        if gate.n_est != len(match.pairs) + match.false_est:
            raise ValueError(f"{activity}: gate n_est {gate.n_est} differs")
        if gate.n_gt != n_gt[activity]:
            raise ValueError(f"{activity}: gate n_gt {gate.n_gt} differs")
        n_pass = sum(bool(_within_pass_limit(abs(e))) for _, _, e in match.pairs)
        if gate.n_pairs < n_pass:
            raise RuntimeError(
                f"{activity}: gate pairs {gate.n_pairs} < ungated passing pairs "
                f"{n_pass} (spec 5-1 property 1)"
            )


def _gate_fields(
    gate_matches: Mapping[str, GateMatch], total_gt: int, total_est: int
) -> dict[str, float | int | None]:
    """Micro-pooled gate metrics (spec 5-1 table); ``None`` where undefined."""
    gate_pairs = sum(gate.n_pairs for gate in gate_matches.values())
    abs_error_sum = sum(gate.abs_error_sum for gate in gate_matches.values())
    return {
        "gate_pairs": gate_pairs,
        "gate_recall": _ratio_or_none(gate_pairs, total_gt),
        "gate_precision": _ratio_or_none(gate_pairs, total_est),
        "gate_f1": _ratio_or_none(2 * gate_pairs, total_gt + total_est),
        "gate_mae_m": _ratio_or_none(abs_error_sum, gate_pairs),
    }


def _tie_fields(
    matches: Mapping[str, MatchResult], ties: Mapping[str, TieRange]
) -> dict[str, int]:
    """Pooled tie range (spec 12-1-1): activities are independent, so the range
    of the scope is the sum of the activity ranges.

    Raises:
        ValueError: ``ties`` does not cover exactly the activities of ``matches``.
        RuntimeError: A ``TieRange.n_pass`` differs from its official count.
    """
    if set(ties) != set(matches):
        raise ValueError("ties must cover exactly the activities of matches")
    for activity, match in matches.items():
        if ties[activity].n_pass != _n_pass(match):
            raise RuntimeError(
                f"{activity}: tie n_pass {ties[activity].n_pass} differs from the "
                f"official pass count {_n_pass(match)}"
            )
    return {
        "tie_pass_min": sum(tie.pass_min for tie in ties.values()),
        "tie_pass_max": sum(tie.pass_max for tie in ties.values()),
        "tie_activities": sum(tie.pass_max > tie.pass_min for tie in ties.values()),
    }


def _observability_fields(
    matches: Mapping[str, MatchResult],
    n_gt: Mapping[str, int],
    observability: Mapping[str, Sequence[str | None]],
) -> dict[str, int]:
    """Unobservable-truth counts (spec 12-2-3) on the official pairs.

    Raises:
        ValueError: ``observability`` does not cover exactly the activities of
            ``matches`` or has the wrong length for an activity.
    """
    if set(observability) != set(matches):
        raise ValueError("observability must cover exactly the activities of matches")
    for activity, verdicts in observability.items():
        if len(verdicts) != n_gt[activity]:
            raise ValueError(
                f"{activity}: {len(verdicts)} verdicts for {n_gt[activity]} truths"
            )
    n_pass_unobservable = 0
    for activity, match in matches.items():
        for _, gt_index, error in match.pairs:
            is_unobservable = observability[activity][gt_index] == OBS_UNOBSERVABLE
            if is_unobservable and _within_pass_limit(abs(error)):
                n_pass_unobservable += 1
    verdicts = [v for activity in matches for v in observability[activity]]
    return {
        "n_gt_unobservable": verdicts.count(OBS_UNOBSERVABLE),
        "n_pass_unobservable": n_pass_unobservable,
        "n_gt_window_unknown": verdicts.count(OBS_UNKNOWN),
    }


def summarize(
    matches: Mapping[str, MatchResult],
    n_gt: Mapping[str, int],
    gate_matches: Mapping[str, GateMatch] | None = None,
    ties: Mapping[str, TieRange] | None = None,
    observability: Mapping[str, Sequence[str | None]] | None = None,
) -> Summary:
    """Pooled pair metrics; pass A per pair, pass B per ground truth (5).

    With ``gate_matches`` (one per activity of ``matches``) the gate fields are
    filled (spec 5-1, auxiliary); without them they stay ``None``. ``ties`` and
    ``observability`` (v1.9, spec 12-1 and 12-2, auxiliary) fill the tie-range
    and unobservable-truth fields the same way; ``observability`` holds one
    verdict per truth of each activity, in truth-index order.

    Raises:
        ValueError: ``gate_matches``, ``ties`` or ``observability`` does not
            describe the same activities (or truths).
        RuntimeError: gate pairs < ungated passing pairs in an activity, or a
            ``TieRange.n_pass`` differs from the official pass count.
    """
    errors = np.array([e for m in matches.values() for _, _, e in m.pairs], dtype=float)
    total_gt = sum(n_gt[a] for a in matches)
    total_est = sum(len(m.pairs) + m.false_est for m in matches.values())
    n_pairs = len(errors)
    absolute = np.abs(errors)
    n_pass = int(np.sum(_within_pass_limit(absolute)))
    auxiliary: dict[str, float | int | None] = {}
    if gate_matches is not None:
        _check_gate_matches(matches, n_gt, gate_matches)
        auxiliary |= _gate_fields(gate_matches, total_gt, total_est)
    if ties is not None:
        auxiliary |= _tie_fields(matches, ties)
    if observability is not None:
        auxiliary |= _observability_fields(matches, n_gt, observability)
    return Summary(
        n_activities=len(matches),
        n_gt=total_gt,
        n_est=total_est,
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
        **auxiliary,
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
    n_gate_pairs: np.ndarray
    """``(trials, activities)`` gate pairs of the same draws (spec 6, v1.6)."""
    n_est: np.ndarray
    """``(activities,)`` estimate count (the observed count, fixed per trial)."""


def draw_baseline(
    n_est_by_act: Mapping[str, int],
    gt_by_act: Mapping[str, Sequence[float]],
    trials: int = BASELINE_TRIALS,
    seed: int = BASELINE_SEED,
    mode: str = "activity",
) -> BaselineDraws:
    """Same number of uniform random depths per activity, same matching (6).

    Draw order is trial-major, activities in ``gt_by_act`` order, from one
    ``default_rng(seed)``. The gate matching (v1.6) reuses each draw and takes
    no extra random numbers, so the ungated counts are unchanged.
    """
    rng = np.random.default_rng(seed)
    activities = tuple(a for a in gt_by_act if a in n_est_by_act)
    shape = (trials, len(activities))
    n_pass = np.zeros(shape, dtype=int)
    n_pairs = np.zeros(shape, dtype=int)
    abs_error_sum = np.zeros(shape, dtype=float)
    n_gate_pairs = np.zeros(shape, dtype=int)
    for trial in range(trials):
        for column, activity in enumerate(activities):
            low, high = baseline_range(gt_by_act[activity], mode)
            draws = rng.uniform(low, high, size=n_est_by_act[activity])
            match = match_activity(list(draws), list(gt_by_act[activity]))
            gate = match_activity_gated(list(draws), list(gt_by_act[activity]))
            errors = np.abs([error for _, _, error in match.pairs])
            n_pass[trial, column] = int(np.sum(_within_pass_limit(errors)))
            n_pairs[trial, column] = len(errors)
            abs_error_sum[trial, column] = float(errors.sum())
            n_gate_pairs[trial, column] = gate.n_pairs
    return BaselineDraws(
        activities=activities,
        n_gt=np.array([len(gt_by_act[a]) for a in activities], dtype=int),
        n_pass=n_pass,
        n_pairs=n_pairs,
        abs_error_sum=abs_error_sum,
        n_gate_pairs=n_gate_pairs,
        n_est=np.array([n_est_by_act[a] for a in activities], dtype=int),
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
        gate_pairs_values=draws.n_gate_pairs[:, columns].sum(axis=1),
        n_est_total=int(draws.n_est[columns].sum()),
        n_gt_total=total_gt,
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


# ----------------------------------------------------------------- duplicate radargrams
# (spec 4-5, v1.7)
#
# The rules use file hashes, line lists and activity numbers only, never a
# ground-truth depth. Depths of every line are still computed; the rules only
# decide which candidates and line counts enter the evaluation.

LineFile = tuple[str, str]
"""``(activity_id, path)`` of one radargram file."""


def activity_order_key(activity_id: str) -> tuple[int, int]:
    """Numeric order of an activity id: ``"010.11"`` -> ``(10, 11)``.

    String order would put ``"010.10"`` before ``"010.9"``.

    Raises:
        ValueError: The id is not ``<digits>.<digits>``.
    """
    site, separator, sub = activity_id.partition(".")
    if not (separator and site.isdecimal() and sub.isdecimal()):
        raise ValueError(f"activity id must look like '010.11', got {activity_id!r}")
    return int(site), int(sub)


def _path_number(path: str) -> int:
    """Trailing integer of a line path (``"Path12"`` -> 12)."""
    match = re.search(r"(\d+)$", path)
    if match is None:
        raise ValueError(f"line path has no trailing number: {path!r}")
    return int(match.group(1))


def _file_label(line_file: LineFile) -> str:
    activity, path = line_file
    return f"{activity}/{path}"


@dataclass(frozen=True)
class DuplicatePair:
    """One row of the duplicate list: two byte-identical radargram files."""

    pair_id: int
    sha256: str
    activity_a: str
    path_a: str
    activity_b: str
    path_b: str
    scope: str
    """``DUP_WITHIN`` or ``DUP_ACROSS``."""

    @property
    def files(self) -> tuple[LineFile, LineFile]:
        return (self.activity_a, self.path_a), (self.activity_b, self.path_b)

    @property
    def within_copy(self) -> LineFile:
        """The file with the higher path number (the representative is the
        lower one, whatever the column order; 4-5-3 item 1)."""
        return max(self.files, key=lambda line_file: _path_number(line_file[1]))


@dataclass(frozen=True)
class DuplicateMark:
    """Candidates csv duplicate columns of one file (spec 8, v1.7)."""

    kind: str
    """``""`` (no mark, also a within-activity representative), ``DUP_WITHIN``
    (a within-activity copy) or ``DUP_ACROSS``."""
    twin: str
    """The other file of the pair, ``<activity_id>/<path>``."""


@dataclass(frozen=True)
class DuplicateTable:
    """The committed duplicate list (``load_duplicate_table``)."""

    pairs: tuple[DuplicatePair, ...] = ()

    def within_copies(self) -> frozenset[LineFile]:
        """Higher-numbered file of every within-activity pair."""
        return frozenset(p.within_copy for p in self.pairs if p.scope == DUP_WITHIN)

    def marks(self) -> dict[LineFile, DuplicateMark]:
        """Both files of every pair -> their csv mark."""
        copies = self.within_copies()
        marks = {}
        for pair in self.pairs:
            first, second = pair.files
            for line_file, twin in ((first, second), (second, first)):
                is_representative = pair.scope == DUP_WITHIN and line_file not in copies
                kind = "" if is_representative else pair.scope
                marks[line_file] = DuplicateMark(kind, _file_label(twin))
        return marks

    def across_activities(self) -> frozenset[str]:
        """Activities with at least one across-activity duplicate line."""
        return frozenset(
            activity
            for pair in self.pairs
            if pair.scope == DUP_ACROSS
            for activity in (pair.activity_a, pair.activity_b)
        )


def _duplicate_pair(row: Mapping[str, str]) -> DuplicatePair:
    return DuplicatePair(
        pair_id=int(row["pair_id"]),
        sha256=row["sha256_npy"].strip().lower(),
        activity_a=row["activity_a"].strip(),
        path_a=row["path_a"].strip(),
        activity_b=row["activity_b"].strip(),
        path_b=row["path_b"].strip(),
        scope=row["scope"].strip(),
    )


def _check_pair_scope(pair: DuplicatePair) -> None:
    """Scope rules of 4-5-2 for one row; raises ``ValueError``."""
    if pair.scope not in DUP_SCOPES:
        raise ValueError(f"pair {pair.pair_id}: unknown scope {pair.scope!r}")
    same_activity = pair.activity_a == pair.activity_b
    if pair.scope == DUP_ACROSS and same_activity:
        raise ValueError(f"pair {pair.pair_id}: {DUP_ACROSS} within one activity")
    if pair.scope == DUP_WITHIN:
        if not same_activity:
            raise ValueError(f"pair {pair.pair_id}: {DUP_WITHIN} across activities")
        if _path_number(pair.path_a) == _path_number(pair.path_b):
            raise ValueError(f"pair {pair.pair_id}: no lower path number to keep")


def _check_duplicate_pairs(pairs: Sequence[DuplicatePair]) -> None:
    """Table rules of 4-5-2: one row per file, one across partner per activity.

    Raises:
        ValueError: The first rule broken, with the pair or file named.
    """
    seen_files: set[LineFile] = set()
    partners: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        _check_pair_scope(pair)
        for line_file in pair.files:
            if line_file in seen_files:
                raise ValueError(f"{_file_label(line_file)} appears in two rows")
            seen_files.add(line_file)
        if pair.scope == DUP_ACROSS:
            partners[pair.activity_a].add(pair.activity_b)
            partners[pair.activity_b].add(pair.activity_a)
    for activity, others in sorted(partners.items()):
        if len(others) > 1:
            # keep_one is undefined when activity pairs overlap (4-5-2).
            raise ValueError(f"{activity} has across partners {sorted(others)}")


def load_duplicate_table(path: Path) -> DuplicateTable:
    """Read and check the duplicate list (``utf-8-sig``, spec 4-5-2).

    Raises:
        ValueError: A required column is missing or a table rule is broken.
    """
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in DUPLICATE_TABLE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    pairs = tuple(_duplicate_pair(row) for row in frame.to_dict("records"))
    _check_duplicate_pairs(pairs)
    return DuplicateTable(pairs)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _run_line_files(lines: pd.DataFrame) -> list[LineFile]:
    """Distinct ``(activity_id, path)`` of the run's line table, in table order."""
    pairs = zip(
        lines["activity_id"].astype(str), lines["path"].astype(str), strict=True
    )
    return list(dict.fromkeys(pairs))


def _listed_pair_problems(
    hashes: Mapping[LineFile, str], table: DuplicateTable
) -> list[str]:
    """Rule 2 of 4-5-2: every run file of a listed pair has the listed hash."""
    problems = []
    for pair in table.pairs:
        present = [f for f in pair.files if f in hashes]
        for line_file in present:
            if hashes[line_file] != pair.sha256:
                problems.append(
                    f"{_file_label(line_file)} hash differs from list pair "
                    f"{pair.pair_id}"
                )
        if len(present) == 2 and hashes[present[0]] != hashes[present[1]]:
            problems.append(
                f"listed pair {pair.pair_id} differs: "
                f"{_file_label(present[0])} vs {_file_label(present[1])}"
            )
    return problems


def _unlisted_group_problems(
    hashes: Mapping[LineFile, str], table: DuplicateTable
) -> list[str]:
    """Rule 3 of 4-5-2: identical run files are exactly the listed pairs."""
    groups: dict[str, list[LineFile]] = defaultdict(list)
    for line_file, digest in hashes.items():
        groups[digest].append(line_file)
    listed = {frozenset(p.files) for p in table.pairs}
    problems = []
    for members in groups.values():
        labels = ", ".join(_file_label(f) for f in members)
        if len(members) > 2:
            problems.append(f"more than two identical files: {labels}")
        elif len(members) == 2 and frozenset(members) not in listed:
            problems.append(f"identical files not in the list: {labels}")
    return problems


def verify_duplicate_hashes(
    root: Path, lines: pd.DataFrame, table: DuplicateTable
) -> tuple[int, float]:
    """Check the committed list against the run's radargram files (4-5-2).

    Reads every run line's ``.npy`` bytes (read-only) and computes SHA-256.

    Returns:
        ``(files hashed, seconds)``.

    Raises:
        RuntimeError: A listed hash differs, a listed pair is not identical, or
            identical files are missing from the list (all problems named).
    """
    started = time.perf_counter()
    hashes = {
        line_file: _file_sha256(
            root / RADARGRAM_DIR / line_file[0] / f"{line_file[1]}.npy"
        )
        for line_file in _run_line_files(lines)
    }
    problems = _listed_pair_problems(hashes, table) + _unlisted_group_problems(
        hashes, table
    )
    if problems:
        raise RuntimeError("duplicate radargram check failed: " + "; ".join(problems))
    return len(hashes), time.perf_counter() - started


def unique_line_counts(lines: pd.DataFrame, table: DuplicateTable) -> dict[str, int]:
    """Lines per activity without the within-activity copies (4-5-3 item 1).

    Every activity of ``lines`` keeps its key, like the plain line count.
    """
    copies = table.within_copies()
    counts = dict.fromkeys(sorted(set(lines["activity_id"].astype(str))), 0)
    for line_file in _run_line_files(lines):
        if line_file not in copies:
            counts[line_file[0]] += 1
    return counts


def drop_within_copies(
    candidates: Sequence[CandidateResult], table: DuplicateTable
) -> list[CandidateResult]:
    """Candidates without those on within-activity copies (they repeat the
    representative's candidates row for row, 4-5-3 item 1)."""
    copies = table.within_copies()
    return [c for c in candidates if (c.activity_id, c.path) not in copies]


@dataclass(frozen=True)
class KeepOneDecision:
    """``keep_one`` choice for one across-activity pair of activities (4-5-4)."""

    activities: tuple[str, str]
    """Both activities in ``activity_order_key`` order."""
    n_shared: int
    n_lines: tuple[int, int]
    """Unique line counts of ``activities`` (same order)."""
    kept: str
    dropped: str
    reason: str
    """``strict_superset``, ``equal_sets`` or ``lower_number``."""
    dropped_paths: tuple[str, ...]
    """Shared lines removed from ``dropped``, by path number."""
    dropped_activity_removed: bool
    """No line of ``dropped`` is left, so it leaves the evaluation."""


def _keep_one_decision(
    activities: tuple[str, str],
    shared_paths: Sequence[tuple[str, str]],
    lines_per_activity: Mapping[str, int],
) -> KeepOneDecision:
    """Strict superset wins; otherwise the lower activity number (4-5-4)."""
    low, high = activities
    n_shared = len(shared_paths)
    n_low, n_high = lines_per_activity[low], lines_per_activity[high]
    if n_shared == n_low < n_high:
        kept, dropped, reason = high, low, KEEP_REASON_SUPERSET
    elif n_shared == n_high < n_low:
        kept, dropped, reason = low, high, KEEP_REASON_SUPERSET
    elif n_shared == n_low == n_high:
        kept, dropped, reason = low, high, KEEP_REASON_EQUAL
    else:
        kept, dropped, reason = low, high, KEEP_REASON_LOWER
    side = 0 if dropped == low else 1
    dropped_paths = tuple(sorted((p[side] for p in shared_paths), key=_path_number))
    return KeepOneDecision(
        activities=activities,
        n_shared=n_shared,
        n_lines=(n_low, n_high),
        kept=kept,
        dropped=dropped,
        reason=reason,
        dropped_paths=dropped_paths,
        dropped_activity_removed=lines_per_activity[dropped] == n_shared,
    )


def keep_one_plan(
    table: DuplicateTable, lines_per_activity: Mapping[str, int]
) -> tuple[KeepOneDecision, ...]:
    """``keep_one`` decisions for the pairs whose two activities are in the run.

    Args:
        table: The duplicate list (one across partner per activity is checked
            on load, so activity pairs do not overlap).
        lines_per_activity: Unique line counts (``unique_line_counts``).
    """
    shared: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for pair in table.pairs:
        if pair.scope != DUP_ACROSS:
            continue
        if not all(a in lines_per_activity for a in (pair.activity_a, pair.activity_b)):
            continue
        low_file, high_file = sorted(
            pair.files, key=lambda line_file: activity_order_key(line_file[0])
        )
        shared[(low_file[0], high_file[0])].append((low_file[1], high_file[1]))
    ordered = sorted(shared, key=lambda acts: activity_order_key(acts[0]))
    return tuple(
        _keep_one_decision(acts, shared[acts], lines_per_activity) for acts in ordered
    )


def apply_dup_handling(
    candidates: Sequence[CandidateResult],
    lines_per_activity: Mapping[str, int],
    table: DuplicateTable,
    handling: str,
) -> tuple[list[CandidateResult], dict[str, int]]:
    """Candidates and line counts of one across-activity handling (4-5-4).

    ``keep_both`` returns copies of the inputs unchanged. Activities removed by
    ``keep_one`` (no line left) or ``drop_all`` leave ``lines_per_activity``,
    so their ground truth is not evaluated either.

    Raises:
        ValueError: ``handling`` is not one of ``DUP_HANDLINGS``.
    """
    if handling == KEEP_BOTH:
        return list(candidates), dict(lines_per_activity)
    if handling == DROP_ALL:
        removed = table.across_activities()
        counts = {a: n for a, n in lines_per_activity.items() if a not in removed}
        return [c for c in candidates if c.activity_id in counts], counts
    if handling != KEEP_ONE:
        raise ValueError(f"unknown duplicate handling {handling!r}")
    counts = dict(lines_per_activity)
    dropped_files: set[LineFile] = set()
    for decision in keep_one_plan(table, lines_per_activity):
        dropped_files |= {(decision.dropped, p) for p in decision.dropped_paths}
        counts[decision.dropped] -= len(decision.dropped_paths)
        if decision.dropped_activity_removed:
            del counts[decision.dropped]
    kept = [
        c
        for c in candidates
        if c.activity_id in counts and (c.activity_id, c.path) not in dropped_files
    ]
    return kept, counts


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
    tie_rows: list[dict[str, object]] = field(default_factory=list)
    """``<prefix>_ties.csv`` rows (12-1-5, v1.9)."""
    unobservable_rows: list[dict[str, object]] = field(default_factory=list)
    """``<prefix>_unobservable.csv`` rows (12-2-4, v1.9)."""


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
            and key not in GATE_SUMMARY_FIELDS
            and key not in AUX_SUMMARY_FIELDS
        },
    }
    row.update(_baseline_columns("baseline", baselines.get("activity"), summary))
    row.update(_baseline_columns("baseline_global", baselines.get("global"), summary))
    # v1.6 gate columns go at the end so every earlier column keeps its position.
    row.update(_observed_gate_columns(summary))
    row.update(
        _baseline_gate_columns(
            "baseline", baselines.get("activity"), summary.gate_pairs
        )
    )
    row.update(
        _baseline_gate_columns(
            "baseline_global", baselines.get("global"), summary.gate_pairs
        )
    )
    # v1.9 columns after the gate columns, for the same reason (spec 9-8 C).
    row.update(_aux_columns(summary))
    return row


def _aux_columns(summary: Summary) -> dict[str, object]:
    """The 11 ``AUX_COLUMNS`` (spec 9-8 C): counts as is, ratios to 4 places,
    blank where undefined."""
    n_gt = summary.n_gt
    tie_min, tie_max = summary.tie_pass_min, summary.tie_pass_max
    n_unobservable = summary.n_gt_unobservable
    n_pass_unobservable = summary.n_pass_unobservable
    n_observable = None if n_unobservable is None else n_gt - n_unobservable
    pass_b_observable = None
    if n_observable is not None and n_pass_unobservable is not None:
        pass_b_observable = _ratio_or_none(
            summary.n_pass - n_pass_unobservable, n_observable
        )
    values: dict[str, float | int | None] = {
        "n_pass": summary.n_pass,
        "tie_pass_min": tie_min,
        "tie_pass_max": tie_max,
        "tie_pass_b_min": None if tie_min is None else _ratio_or_none(tie_min, n_gt),
        "tie_pass_b_max": None if tie_max is None else _ratio_or_none(tie_max, n_gt),
        "tie_activities": summary.tie_activities,
        "n_gt_unobservable": n_unobservable,
        "n_pass_unobservable": n_pass_unobservable,
        "n_gt_window_unknown": summary.n_gt_window_unknown,
        "n_gt_observable": n_observable,
        "pass_b_observable": pass_b_observable,
    }
    return {
        name: value if isinstance(value, int) else _finite_rounded_or_blank(value, 4)
        for name, value in values.items()
    }


def _observed_gate_columns(summary: Summary) -> dict[str, object]:
    """Observed gate csv columns (spec 8, v1.6): count as is, ratios to 4 places."""
    return {name: _rounded(getattr(summary, name), 4) for name in GATE_SUMMARY_FIELDS}


def _gate_rates_per_trial(stats: BaselineStats) -> dict[str, np.ndarray | None]:
    """Per-trial recall, precision and F1 of the baseline; ``None`` if undefined."""
    pairs = stats.gate_pairs_values
    n_gt, n_est = stats.n_gt_total, stats.n_est_total
    return {
        "recall": pairs / n_gt if n_gt else None,
        "precision": pairs / n_est if n_est else None,
        "f1": 2 * pairs / (n_gt + n_est) if n_gt + n_est else None,
    }


def _mean_and_interval(per_trial: np.ndarray | None) -> tuple[float | None, ...]:
    """Mean, 2.5 % and 97.5 % over the trials (the existing baseline columns' way)."""
    if per_trial is None:
        return None, None, None
    return (
        float(np.nanmean(per_trial)),
        float(np.nanpercentile(per_trial, 2.5)),
        float(np.nanpercentile(per_trial, 97.5)),
    )


def _gate_position(stats: BaselineStats, gate_pairs: int) -> tuple[float, float]:
    """One-sided ``p`` (K_b >= K) and mid-rank percentile of the observed K (spec 6)."""
    trial_pairs = stats.gate_pairs_values
    p_value = float(np.mean(trial_pairs >= gate_pairs))
    percentile = 100 * float(
        np.mean(trial_pairs < gate_pairs) + 0.5 * np.mean(trial_pairs == gate_pairs)
    )
    return p_value, percentile


def _baseline_gate_columns(
    prefix: str, stats: BaselineStats | None, gate_pairs: int | None
) -> dict[str, object]:
    """Baseline gate csv columns (spec 6 and 8, v1.6).

    ``baseline``: recall, precision and F1 (mean, 2.5 and 97.5 %), ``p`` and
    ``pct``. ``baseline_global``: F1 (mean, 2.5 and 97.5 %) and ``p``. Blank
    where a denominator is 0, without a baseline, or (``p``, ``pct`` only)
    without an observed gate count.
    """
    metrics = ("recall", "precision", "f1") if prefix == "baseline" else ("f1",)
    rates = {} if stats is None else _gate_rates_per_trial(stats)
    values: dict[str, float | None] = {}
    for metric in metrics:
        mean, p025, p975 = _mean_and_interval(rates.get(metric))
        values |= {
            f"gate_{metric}_mean": mean,
            f"gate_{metric}_p025": p025,
            f"gate_{metric}_p975": p975,
        }
    p_value = percentile = None
    if stats is not None and gate_pairs is not None:
        p_value, percentile = _gate_position(stats, gate_pairs)
    values["gate_p"] = p_value
    if prefix == "baseline":
        values["gate_pct"] = percentile
    return {f"{prefix}_{name}": _rounded(value, 4) for name, value in values.items()}


def _baseline_columns(
    prefix: str, stats: BaselineStats | None, summary: Summary
) -> dict[str, object]:
    """Baseline csv columns (spec 8, v1.2).

    The activity-range baseline reports MAE, pass A and pass B; the global
    range only pass B and ``p``. Blank when there is no baseline or the value is
    undefined (v1.7, user decision b: with no estimate in the scope every trial
    has 0 pairs, so the MAE and pass A means are NaN and are written blank).
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
        f"{prefix}_{name}": _finite_rounded_or_blank(value, 4)
        for name, value in values.items()
    }


def _finite_rounded_or_blank(value: float | None, digits: int) -> object:
    return "" if value is None or not math.isfinite(value) else round(value, digits)


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
    The record windows of the evaluated lines (12-2) come from ``candidates``.
    """
    evaluation = Evaluation(gt_activities=frozenset(gt_sets["exclude_x"]))
    windows = activity_windows(candidates)
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
                        evaluation,
                        estimates,
                        ground_truth,
                        combination,
                        args,
                        windows=windows,
                    )
    _sort_auxiliary_rows(evaluation, list(gt_sets))
    return evaluation


def activity_windows(
    candidates: Sequence[object],
) -> dict[str, list[RecordWindow]]:
    """One record window per evaluated line ``(activity_id, path)`` (12-2-1).

    Every candidate counts whatever its status or QC (the window is a line
    property); candidates without a ``record_window`` give none.
    """
    by_line: dict[tuple[str, str], RecordWindow] = {}
    for candidate in candidates:
        window = getattr(candidate, "record_window", None)
        if window is not None:
            by_line.setdefault((candidate.activity_id, candidate.path), window)
    windows: dict[str, list[RecordWindow]] = defaultdict(list)
    for (activity, _), window in by_line.items():
        windows[activity].append(window)
    return dict(windows)


def _sort_auxiliary_rows(evaluation: Evaluation, gt_set_order: Sequence[str]) -> None:
    """Row order of the ties and unobservable csvs (12-1-5, 12-2-4)."""

    def common_key(row: Mapping[str, object]) -> tuple:
        return (
            ZERO_VARIANTS.index(str(row["zero_variant"])),
            gt_set_order.index(str(row["gt_set"])),
            ASSUMPTIONS.index(str(row["assumption"])),
            activity_order_key(str(row["activity_id"])),
        )

    evaluation.tie_rows.sort(
        key=lambda row: (VARIANTS.index(str(row["variant"])), *common_key(row))
    )
    evaluation.unobservable_rows.sort(
        key=lambda row: (*common_key(row), int(str(row["gt_index"])))
    )


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
                    "within_30cm": _within_pass_limit(abs(error)),
                }
            )
    return rows


def _evaluate_one(
    evaluation: Evaluation,
    estimates: Mapping[str, list[float]],
    ground_truth: Mapping[str, list[GtItem]],
    combination: Combination,
    args: argparse.Namespace,
    trials: Mapping[str, int] | None = None,
    windows: Mapping[str, Sequence[RecordWindow]] | None = None,
) -> None:
    """Match, summarize and baseline one combination into ``evaluation``.

    ``trials`` (v1.7) overrides the baseline trials per mode; ``None`` keeps
    ``baseline_trials_for(combination.variant, args)``. ``windows`` (v1.9) are
    the record windows per activity for the unobservable truths (12-2); an
    activity without windows has only ``window_unknown`` truths.
    """
    if trials is None:
        trials = baseline_trials_for(combination.variant, args)
    windows = windows or {}
    gt_depths = {
        a: [item.depth_for(combination.assumption) for item in items]
        for a, items in ground_truth.items()
        if a in estimates
    }
    matches = {a: match_activity(estimates[a], gt_depths[a]) for a in gt_depths}
    gate_matches = {
        a: match_activity_gated(estimates[a], gt_depths[a]) for a in gt_depths
    }
    # The official matches are reused as they are: no re-solve (spec 9-8 item 1).
    ties = {a: tie_range(estimates[a], gt_depths[a], matches[a]) for a in gt_depths}
    observability = {
        a: [
            truth_observability(depth, windows.get(a, ()), combination.zero_variant)
            for depth in gt_depths[a]
        ]
        for a in gt_depths
    }
    evaluation.match_rows.extend(
        _match_rows(matches, ground_truth, gt_depths, combination)
    )
    evaluation.tie_rows.extend(_tie_rows(ties, estimates, gt_depths, combination))
    if combination.variant == "main":
        evaluation.unobservable_rows.extend(
            _unobservable_rows(
                observability, matches, ground_truth, gt_depths, combination, windows
            )
        )
    n_gt = {a: len(v) for a, v in gt_depths.items()}
    n_est = {a: len(estimates[a]) for a in matches}
    # One draw per baseline mode; every scope is aggregated from these trials.
    draws = {
        mode: draw_baseline(n_est, gt_depths, mode_trials, args.baseline_seed, mode)
        for mode, mode_trials in trials.items()
    }
    sites = sorted({site_of_activity(a) for a in matches})
    scopes = [("all", list(matches))] + [
        (f"site_{s}", [a for a in matches if site_of_activity(a) == s]) for s in sites
    ]
    for scope, activities in scopes:
        subset = {a: matches[a] for a in activities}
        gate_subset = {a: gate_matches[a] for a in activities}
        baselines = {
            mode: baseline_stats(draw, activities) for mode, draw in draws.items()
        }
        summary = summarize(
            subset,
            n_gt,
            gate_subset,
            ties={a: ties[a] for a in activities},
            observability={a: observability[a] for a in activities},
        )
        evaluation.summary_rows.append(
            _summary_row(scope, combination, summary, baselines)
        )


def _tie_rows(
    ties: Mapping[str, TieRange],
    estimates: Mapping[str, Sequence[float]],
    gt_depths: Mapping[str, Sequence[float]],
    combination: Combination,
) -> list[dict[str, object]]:
    """Ties csv rows: activities whose pass count changes with the tie (12-1-5)."""
    return [
        {
            "variant": combination.variant,
            "zero_variant": combination.zero_variant,
            "gt_set": combination.gt_set,
            "assumption": combination.assumption,
            "activity_id": activity,
            "site": site_of_activity(activity),
            "n_est": len(estimates[activity]),
            "n_gt": len(gt_depths[activity]),
            "n_pass": tie.n_pass,
            "tie_pass_min": tie.pass_min,
            "tie_pass_max": tie.pass_max,
        }
        for activity, tie in ties.items()
        if tie.pass_max > tie.pass_min
    ]


def _passing_truths(match: MatchResult) -> set[int]:
    return {
        gt_index for _, gt_index, error in match.pairs if _within_pass_limit(abs(error))
    }


def _largest_depth_limit(
    windows: Sequence[RecordWindow], zero_variant: str
) -> tuple[float | None, int]:
    """Largest ``depth_limit_m`` over the usable windows and their count."""
    limits = []
    n_usable = 0
    for window in windows:
        if _usable_zero(window, zero_variant) is None:
            continue
        n_usable += 1
        limit = depth_limit_m(window, zero_variant)
        if limit is not None:
            limits.append(limit)
    return (max(limits) if limits else None), n_usable


def _unobservable_rows(
    observability: Mapping[str, Sequence[str | None]],
    matches: Mapping[str, MatchResult],
    ground_truth: Mapping[str, list[GtItem]],
    gt_depths: Mapping[str, Sequence[float]],
    combination: Combination,
    windows: Mapping[str, Sequence[RecordWindow]],
) -> list[dict[str, object]]:
    """Unobservable csv rows of one ``main`` combination (12-2-4)."""
    rows = []
    for activity, verdicts in observability.items():
        limit_m, n_windows = _largest_depth_limit(
            windows.get(activity, ()), combination.zero_variant
        )
        passing = _passing_truths(matches[activity])
        for gt_index, verdict in enumerate(verdicts):
            if verdict is None:
                continue
            rows.append(
                {
                    "zero_variant": combination.zero_variant,
                    "gt_set": combination.gt_set,
                    "assumption": combination.assumption,
                    "activity_id": activity,
                    "site": site_of_activity(activity),
                    "gt_index": gt_index,
                    "gt_depth_m": round(gt_depths[activity][gt_index], 4),
                    "gt_code": ground_truth[activity][gt_index].code,
                    "observability": verdict,
                    "depth_limit_m": _rounded(limit_m, 4),
                    "n_windows": n_windows,
                    "pass_main": gt_index in passing,
                }
            )
    return rows


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
                                "within_30cm_eps8": _within_pass_limit(abs(error)),
                            }
                        )
    return rows, summaries


def _main_variant_summary_rows(
    candidates: Sequence[CandidateResult],
    gt_sets: Mapping[str, Mapping[str, list[GtItem]]],
    lines_per_activity: Mapping[str, int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Summary rows of the main path only, activity baseline with aux trials.

    Same order as the ``main`` rows of ``evaluate_combinations`` (zero ->
    gt set -> assumption -> scope); matches and clusters are discarded.
    """
    scratch = Evaluation()
    trials = {"activity": args.baseline_trials_aux}
    windows = activity_windows(candidates)  # the handled lines only (12-2-1)
    for zero_variant in ZERO_VARIANTS:
        estimates, _ = estimates_by_activity(
            candidates,
            zero_variant,
            "main",
            lines_per_activity,
            width_m=args.cluster_width_m,
            linkage=args.cluster_linkage,
            min_lines=args.min_support_lines,
            apex_qc=apex_qc_enabled(args),
        )
        for gt_set, ground_truth in gt_sets.items():
            for assumption in ASSUMPTIONS:
                combination = Combination(gt_set, assumption, zero_variant, "main")
                _evaluate_one(
                    scratch,
                    estimates,
                    ground_truth,
                    combination,
                    args,
                    trials=trials,
                    windows=windows,
                )
    return scratch.summary_rows


def duplicate_sensitivity(
    candidates: Sequence[CandidateResult],
    gt_sets: Mapping[str, Mapping[str, list[GtItem]]],
    lines_per_activity: Mapping[str, int],
    table: DuplicateTable,
    args: argparse.Namespace,
    summary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rows of ``<prefix>_dup_sensitivity.csv`` (spec 4-5-4 and 8; never judged).

    ``keep_both`` copies the summary ``main`` rows unchanged. ``keep_one`` and
    ``drop_all`` rerun the main path (QC rule, clustering, support filter,
    matching, gate) on their candidates and line counts; depths are not
    recomputed.

    Args:
        candidates: Evaluated candidates (within-activity copies dropped).
        lines_per_activity: Unique line counts (``unique_line_counts``).
        summary_rows: ``evaluate_combinations`` summary rows; not modified.
    """
    rows = [
        {"dup_handling": KEEP_BOTH, **row}
        for row in summary_rows
        if row["variant"] == "main"
    ]
    for handling in (KEEP_ONE, DROP_ALL):
        handled, line_counts = apply_dup_handling(
            candidates, lines_per_activity, table, handling
        )
        rows.extend(
            {"dup_handling": handling, **row}
            for row in _main_variant_summary_rows(handled, gt_sets, line_counts, args)
        )
    return rows


# ----------------------------------------------------------------- outputs


def candidate_row(
    candidate: CandidateResult,
    lines: Mapping[tuple[str, str], Mapping[str, object]],
    duplicates: Mapping[LineFile, DuplicateMark] | None = None,
) -> dict[str, object]:
    """One candidates csv row (spec 8).

    ``flags``: depth flags, ``long_line``, ``zero_h_implausible`` (v1.7), then the
    QC flags; the last two never enter ``DepthResult.flags``. ``duplicates``
    (``DuplicateTable.marks``) fills the two v1.7 columns at the end of the row;
    without it they are blank.
    """
    result = candidate.result
    est = result.alt_depths.get(META_ZERO_KEYS["est"])
    act = result.alt_depths.get(META_ZERO_KEYS["act"])
    qc = result.eps_r_refraction_qc_range
    line = lines.get((candidate.activity_id, candidate.path), {})
    flags = list(result.flags)
    if float(line.get("length_m", 0) or 0) >= LONG_LINE_M:
        flags.append("long_line")
    if candidate.zero_h_implausible:
        flags.append(FLAG_ZERO_H)
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
        **_duplicate_columns(candidate, duplicates or {}),
    }


def _duplicate_columns(
    candidate: CandidateResult, duplicates: Mapping[LineFile, DuplicateMark]
) -> dict[str, object]:
    """``duplicate_radargram`` and ``duplicate_of`` (spec 8, v1.7)."""
    mark = duplicates.get((candidate.activity_id, candidate.path))
    if mark is None:
        return dict.fromkeys(DUPLICATE_CSV_COLUMNS, "")
    return {"duplicate_radargram": mark.kind, "duplicate_of": mark.twin}


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


def write_csv_with_header(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    """Like ``write_csv``, but the header is written even without rows (v1.9)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
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


def _gate_section_lines(evaluation: Evaluation, args: argparse.Namespace) -> list[str]:
    """md gate section between "X 포함" and "현장별" (spec 8, v1.6; auxiliary)."""
    overall = [
        r
        for r in evaluation.summary_rows
        if r["scope"] == "all" and r["gt_set"] == "exclude_x"
    ]
    main_rows = [r for r in overall if r["variant"] == "main"]
    detected = {
        (r["variant"], r["assumption"]): r
        for r in overall
        if r["zero_variant"] == "detected"
    }
    variant_rows = [
        detected[(variant, assumption)]
        for variant in variants_for(args)
        for assumption in ASSUMPTIONS
    ]
    return [
        "## 게이트 보조 지표 (±0.30 m, v1.6 — 판정·U-3에 쓰지 않음)",
        "",
        "- 정의: 활동마다 |e| ≤ 0.30 m 짝만 허용해 짝 수를 최대화하고, 같으면 "
        "Σ|e|를 최소화한 매칭(§5-1). 재현율 = 짝/정답, 정밀도 = 짝/추정, "
        "F1 = 2·짝/(정답+추정), 활동 합산. 정답 없는 활동은 넣지 않는다.",
        "- 기준선: 위 표와 같은 시행(같은 추정 수·같은 난수)의 게이트 짝 수. "
        "p = 기준선 짝 수 ≥ 관측 짝 수인 시행 비율, 백분위 = 중간 순위. "
        "재현율·정밀도·F1의 p와 백분위는 같다.",
        "- 판정은 위 '주 결과' 표(게이트 없는 헝가리안, 판정률 A·B)로만 한다. "
        "게이트 재현율 ≥ 판정률 B는 정의상 성립하고 실행 중 활동마다 검사한다.",
        "",
        "### 주 결과 (`main`, 전체, X 제외)",
        "",
        *_md_table(main_rows, GATE_MAIN_TABLE_COLUMNS),
        "",
        "### 변형별 (검출 영점, 전체, X 제외)",
        "",
        *_md_table(variant_rows, GATE_VARIANT_TABLE_COLUMNS),
        "",
    ]


@dataclass(frozen=True)
class NearApexStops:
    """Wings that stopped within ``PREREG_NEAR_APEX_STEPS`` traces (spec 9-8 G)."""

    n_candidates: int
    """``ok`` candidates with a QC (the md section 8 denominator N)."""
    n_any: int
    """Candidates with at least one such wing."""
    n_both: int
    n_wings: int
    """``2 N``: untracked wings stay in the denominator."""
    n_wings_near: int
    reasons: Mapping[str, int]
    """Stop reason (``TrackStop`` value) of the counted wings, by name."""


def _stopped_near_apex(n_picks: int, stop: TrackStop | None) -> bool:
    """Tracking walks one trace per step and stops at the first failed step, so
    ``n_picks`` = stop step - 1 (spec 9-8 G)."""
    return stop is not None and n_picks <= PREREG_NEAR_APEX_STEPS - 1


def near_apex_stop_counts(candidates: Sequence[object]) -> NearApexStops:
    """Near-apex stop counts over the ``ok`` candidates with a QC (spec 9-8 G)."""
    n_candidates = n_any = n_both = 0
    reasons: Counter[str] = Counter()
    for candidate in candidates:
        qc = getattr(candidate, "qc", None)
        if candidate.result.status != "ok" or qc is None:
            continue
        n_candidates += 1
        curvature = qc.curvature
        wings = (
            (curvature.n_left, curvature.stop_left),
            (curvature.n_right, curvature.stop_right),
        )
        near_stops = [stop for n, stop in wings if _stopped_near_apex(n, stop)]
        n_any += bool(near_stops)
        n_both += len(near_stops) == len(wings)
        reasons.update(stop.value for stop in near_stops)
    return NearApexStops(
        n_candidates=n_candidates,
        n_any=n_any,
        n_both=n_both,
        n_wings=2 * n_candidates,
        n_wings_near=sum(reasons.values()),
        reasons=dict(sorted(reasons.items())),
    )


def _percent_phrase(numerator: int, denominator: int) -> str:
    if not denominator:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator}({100 * numerator / denominator:.1f} %)"


def _near_apex_lines(
    candidates: Sequence[CandidateResult], args: argparse.Namespace
) -> list[str]:
    """Last subsection of the v1.9 auxiliary section (spec 9-8 G)."""
    if not apex_qc_enabled(args):
        return ["- 꼭짓점 4 트레이스 안 정지: `apex_qc=off`(계산 안 함)", ""]
    stops = near_apex_stop_counts(candidates)
    reasons = ", ".join(f"`{name}` {n}" for name, n in stops.reasons.items())
    return [
        f"### 꼭짓점 {PREREG_NEAR_APEX_STEPS} 트레이스 안 정지 "
        "(`ok` 후보, 사전 등록 §4 항목 6 — 판정 미사용)",
        "",
        f"- 후보 기준(한쪽 이상): {_percent_phrase(stops.n_any, stops.n_candidates)}, "
        f"양쪽: {stops.n_both}",
        f"- 날개 기준: {_percent_phrase(stops.n_wings_near, stops.n_wings)}",
        f"- 정지 사유(날개): {reasons or '없음'}",
        "",
    ]


def _tie_activity_phrase(evaluation: Evaluation) -> str:
    rows = [
        r
        for r in evaluation.tie_rows
        if (r["variant"], r["zero_variant"], r["assumption"], r["gt_set"])
        == ("main", "detected", "top", "exclude_x")
    ]
    phrases = [
        f"{r['activity_id']} {r['tie_pass_min']}–{r['tie_pass_max']}"  # noqa: RUF001
        f"(판정 {r['n_pass']})"
        for r in rows
    ]
    return ", ".join(phrases) or "없음"


def _unobservable_phrase(evaluation: Evaluation) -> str:
    rows = [
        r
        for r in evaluation.unobservable_rows
        if (r["zero_variant"], r["assumption"], r["gt_set"], r["observability"])
        == ("detected", "top", "exclude_x", OBS_UNOBSERVABLE)
    ]
    phrases = [
        f"{r['activity_id']} {r['gt_depth_m']} m(창 한계 {r['depth_limit_m']} m)"
        for r in rows
    ]
    return ", ".join(phrases) or "없음"


def _auxiliary_v19_section_lines(
    evaluation: Evaluation,
    candidates: Sequence[CandidateResult],
    args: argparse.Namespace,
) -> list[str]:
    """md section between the gate section and "현장별" (spec 12-1-5, 9-8 G)."""
    main_rows = [
        r
        for r in evaluation.summary_rows
        if r["variant"] == "main" and r["scope"] == "all" and r["gt_set"] == "exclude_x"
    ]
    return [
        "## 보조: 동률 범위·관측 불가 정답 (v1.9, M2-10 — 판정·U-3에 쓰지 않음)",
        "",
        "- 동률 범위: 게이트 없는 헝가리안(§5)의 최소합 배정이 여럿일 때"
        "(합 차 ≤ 1e-9 m) 그 배정들의 통과 수 최소·최대. 판정값은 해법기 기본 해"
        "(`n_pass`)이고 바뀌지 않는다.",
        "- 관측 불가 정답: 메타 v로 바꾼 정답 시각이 평가에 들어간 모든 측선에서 "
        "기록 끝 영역(D 규칙, 2.5 주기) 안인 정답. 분모는 그대로 두고, 그것을 뺀 "
        "판정률을 보조로 낸다(공식 짝 그대로, 다시 매칭하지 않음).",
        "- 이 절의 값은 결과를 본 뒤 만든 사후 보조 출력이다(이미 본 자료).",
        "",
        "### 동률 범위 (`main`, 전체, X 제외)",
        "",
        *_md_table(main_rows, TIE_TABLE_COLUMNS),
        "",
        "- 통과 수가 동률로 달라지는 활동(`main`, 검출 영점, 상단, X 제외): "
        + _tie_activity_phrase(evaluation),
        "",
        "### 관측 불가 정답 (기록 끝 아래, `main`, 전체, X 제외)",
        "",
        *_md_table(main_rows, UNOBSERVABLE_TABLE_COLUMNS),
        "",
        "- 관측 불가 정답(검출 영점, 상단, X 제외): "
        + _unobservable_phrase(evaluation),
        "",
        *_near_apex_lines(candidates, args),
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


@dataclass(frozen=True)
class DuplicateReport:
    """Duplicate-radargram facts of one run for the md (spec 8 v1.7 (i), (ii))."""

    table: DuplicateTable
    list_label: str
    """List path as written in md item 1 (project-relative when possible)."""
    decisions: tuple[KeepOneDecision, ...]
    sensitivity_rows: tuple[Mapping[str, object], ...]
    copy_lines: tuple[LineFile, ...]
    """Within-activity copies among the run's lines."""
    n_copy_candidates: int
    """Candidates on ``copy_lines`` (in the csv, not evaluated)."""
    gt_counts: Mapping[str, int]
    """``exclude_x`` ground-truth items per activity."""
    hashed_files: int
    hash_seconds: float


def _line_sort_key(line_file: LineFile) -> tuple[tuple[int, int], int, str]:
    """Activity number, then path number (paths without one first), then name."""
    activity, path = line_file
    match = re.search(r"(\d+)$", path)
    return activity_order_key(activity), int(match.group(1)) if match else -1, path


def _candidate_count_line(
    candidates: Sequence[CandidateResult],
    evaluation: Evaluation,
    duplicates: DuplicateReport | None,
) -> str:
    """md item 1 candidate line; with copies it names the evaluated subset."""
    no_gt = (
        f"정답 없는 활동 {len(evaluation.no_gt_counts)}개"
        f"(주 결과 추정 {sum(evaluation.no_gt_counts.values())}개, 오검출로 세지 않음)"
    )
    if duplicates is None or not duplicates.copy_lines:
        return f"- 꼭짓점 후보 {len(candidates)}개, {no_gt}"
    total = len(candidates) + duplicates.n_copy_candidates
    return (
        f"- 꼭짓점 후보 {total}개(평가 {len(candidates)}개 — 활동 안 사본 측선 "
        f"{len(duplicates.copy_lines)}개의 후보 {duplicates.n_copy_candidates}개 "
        f"제외, csv에는 있음), {no_gt}"
    )


def _duplicate_list_lines(duplicates: DuplicateReport | None) -> list[str]:
    """md item 1 line on the committed list and the hash check (v1.7)."""
    if duplicates is None:
        return []
    return [
        f"- 중복 레이더그램: 목록 `{duplicates.list_label}` "
        f"{len(duplicates.table.pairs)}쌍, 실행 해시 대조 일치"
        f"(파일 {duplicates.hashed_files}개, {duplicates.hash_seconds:.1f} s)"
    ]


def _within_copy_line(duplicates: DuplicateReport) -> str:
    if not duplicates.copy_lines:
        return "- 활동 안 사본: 없음"
    marks = duplicates.table.marks()
    copies = ", ".join(_file_label(f) for f in duplicates.copy_lines)
    representatives = ", ".join(marks[f].twin for f in duplicates.copy_lines)
    return (
        f"- 활동 안 사본: {copies}(대표 {representatives}) — 평가·지지 필터·측선 "
        f"통계에서 뺌, 후보 {duplicates.n_copy_candidates}개"
    )


def _across_line(duplicates: DuplicateReport, evaluation: Evaluation) -> str:
    """Across-activity pairs whose two activities are in the run (4-5-3 item 2)."""
    main_counts = evaluation.estimate_counts.get("main", {})
    pairs = [
        p
        for p in duplicates.table.pairs
        if p.scope == DUP_ACROSS
        and p.activity_a in main_counts
        and p.activity_b in main_counts
    ]
    activities = {a for p in pairs for a in (p.activity_a, p.activity_b)}
    gt_activities = [a for a in activities if a in duplicates.gt_counts]
    n_gt = sum(duplicates.gt_counts[a] for a in gt_activities)
    n_est = sum(main_counts[a] for a in activities)
    return (
        f"- 활동 간 중복: {len(pairs)}쌍, 활동 {len(activities)}개(정답 활동 "
        f"{len(gt_activities)}개, `exclude_x` 정답 {n_gt}개, 주 결과 추정 {n_est}개)"
        " — 주 결과는 둘 다 유지"
    )


def _keep_one_line(decisions: Sequence[KeepOneDecision]) -> str:
    def dropped_phrase(decision: KeepOneDecision) -> str:
        if decision.dropped_activity_removed:
            return f"{decision.dropped} 전체"
        return f"{decision.dropped} {','.join(decision.dropped_paths)}"

    choices = [f"{d.kept} ← {dropped_phrase(d)}({d.reason})" for d in decisions]
    removed = [d.dropped for d in decisions if d.dropped_activity_removed]
    return (
        f"- `keep_one` 남김/뺌: {', '.join(choices) or '없음'}; "
        f"평가에서 빠진 활동 {', '.join(removed) or '없음'}"
    )


def _drop_all_line(duplicates: DuplicateReport, evaluation: Evaluation) -> str:
    run_activities = set(evaluation.estimate_counts.get("main", {}))
    removed = duplicates.table.across_activities() & run_activities
    return f"- `drop_all`: 활동 {len(removed)}개 제외"


def _difference(
    value: object, reference: object, scale: float, digits: int
) -> float | None:
    """``scale * (value - reference)`` rounded; ``None`` when either is missing."""
    if not isinstance(value, int | float) or not isinstance(reference, int | float):
        return None
    return round(scale * (value - reference), digits)


def _overall_detected_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    """Sensitivity rows of (all, exclude_x, detected) by (handling, assumption)."""
    return {
        (str(r["dup_handling"]), str(r["assumption"])): r
        for r in rows
        if r["scope"] == "all"
        and r["gt_set"] == "exclude_x"
        and r["zero_variant"] == "detected"
    }


def _sensitivity_table_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """md sensitivity table: three handlings x two assumptions, ``d_*`` against
    ``keep_both`` of the same assumption (blank in the ``keep_both`` rows)."""
    selected = _overall_detected_rows(rows)
    table_rows = []
    for handling in DUP_HANDLINGS:
        for assumption in ASSUMPTIONS:
            row = selected.get((handling, assumption))
            reference = selected.get((KEEP_BOTH, assumption))
            if row is None or reference is None:
                continue
            deltas: dict[str, object] = {"d_pass_b_pp": "", "d_mae_m": ""}
            if handling != KEEP_BOTH:
                d_pass_b = _difference(
                    row["pass_b"], reference["pass_b"], scale=100, digits=2
                )
                d_mae = _difference(row["mae_m"], reference["mae_m"], scale=1, digits=4)
                deltas = {
                    "d_pass_b_pp": "" if d_pass_b is None else d_pass_b,
                    "d_mae_m": "" if d_mae is None else d_mae,
                }
            table_rows.append({**row, **deltas})
    return table_rows


def _caveat_hits(rows: Sequence[Mapping[str, object]]) -> list[tuple[str, float]]:
    """``(handling/zero/assumption, delta pass B in %p)`` with ``|delta| >= T``.

    Deltas come from the csv values (4 places) of ``all``, ``exclude_x``, over
    3 zeros x 2 assumptions (spec 4-5-5); the boundary is included.
    """
    overall = [r for r in rows if r["scope"] == "all" and r["gt_set"] == "exclude_x"]
    references = {
        (r["zero_variant"], r["assumption"]): r
        for r in overall
        if r["dup_handling"] == KEEP_BOTH
    }
    hits = []
    for row in overall:
        reference = references.get((row["zero_variant"], row["assumption"]))
        if row["dup_handling"] == KEEP_BOTH or reference is None:
            continue
        pass_b, reference_pass_b = row["pass_b"], reference["pass_b"]
        if not isinstance(pass_b, float | int) or not isinstance(
            reference_pass_b, float | int
        ):
            continue
        delta_pp = 100 * (pass_b - reference_pass_b)
        if abs(delta_pp) >= DUP_CAVEAT_PASS_B_PP - DUP_CAVEAT_TOL_PP:
            label = f"{row['dup_handling']}/{row['zero_variant']}/{row['assumption']}"
            hits.append((label, delta_pp))
    return hits


def _caveat_line(rows: Sequence[Mapping[str, object]]) -> str:
    prefix = (
        "- 단서 규칙(§4-5-5, 영점 3 × 가정 2, |Δ판정률 B| ≥ "  # noqa: RUF001
        f"{DUP_CAVEAT_PASS_B_PP:g} %p): "
    )
    hits = _caveat_hits(rows)
    if not hits:
        return prefix + "해당 없음"
    largest = max(abs(delta) for _, delta in hits)
    listed = ", ".join(f"{label} {delta:+.2f} %p" for label, delta in hits)
    return f"{prefix}해당 — {listed}(최대 {largest:.2f} %p)"


def _duplicate_section_lines(
    duplicates: DuplicateReport | None, evaluation: Evaluation
) -> list[str]:
    """md duplicate section between "상태" and "현장 9" (spec 8 v1.7 (ii))."""
    if duplicates is None:
        return []
    return [
        "## 중복 레이더그램 (v1.7, 원 데이터셋 자체 중복 — 민감도는 판정에 쓰지 않음)",
        "",
        _within_copy_line(duplicates),
        _across_line(duplicates, evaluation),
        _keep_one_line(duplicates.decisions),
        _drop_all_line(duplicates, evaluation),
        "",
        "### 민감도 (주 결과 규칙, 검출 영점, 전체, X 제외)",
        "",
        *_md_table(
            _sensitivity_table_rows(duplicates.sensitivity_rows),
            DUP_SENSITIVITY_TABLE_COLUMNS,
        ),
        "",
        _caveat_line(duplicates.sensitivity_rows),
        "",
    ]


def _line_groups(
    candidates: Sequence[CandidateResult],
) -> dict[LineFile, list[CandidateResult]]:
    groups: dict[LineFile, list[CandidateResult]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.activity_id, candidate.path)].append(candidate)
    return groups


def _zero_flag_counts(
    candidates: Sequence[CandidateResult], gt_activities: frozenset[str]
) -> tuple[dict[int, Counter[str]], list[LineFile]]:
    """Per-site counts of the zero-flag table and the flagged lines (spec 7-1).

    Only lines with candidates are seen (the evaluation loads no other line).
    ``main_input`` = ``ok`` and not dropped by the main QC rule (B+D+E).
    """
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    flagged_lines = []
    for line_file, members in _line_groups(candidates).items():
        site_counts = counts[site_of_activity(line_file[0])]
        site_counts["lines"] += 1
        rules = {m.result.zero_rule for m in members}
        site_counts["surface_late"] += ZeroRule.AIR_SURFACE_LATE in rules
        if not any(m.zero_h_implausible for m in members):
            continue
        flagged_lines.append(line_file)
        ok = [m for m in members if m.result.status == "ok"]
        site_counts["flagged"] += 1
        site_counts["candidates"] += len(members)
        site_counts["ok"] += len(ok)
        site_counts["main_input"] += sum(not _is_qc_excluded(m, "main") for m in ok)
        site_counts["flagged_gt_lines"] += line_file[0] in gt_activities
    return counts, sorted(flagged_lines, key=_line_sort_key)


def _zero_flag_lines(
    candidates: Sequence[CandidateResult], evaluation: Evaluation
) -> list[str]:
    """md subsection of the site 9 section (spec 8 v1.7 (iii)); always written."""
    counts, flagged_lines = _zero_flag_counts(candidates, evaluation.gt_activities)
    count_columns = ZERO_FLAG_TABLE_COLUMNS[1:]
    rows: list[dict[str, object]] = [
        {"site": site, **{c: site_counts[c] for c in count_columns}}
        for site, site_counts in sorted(counts.items())
        if site_counts["surface_late"] > 0
    ]
    totals = {c: sum(int(row[c]) for row in rows) for c in count_columns}
    rows.append({"site": "합계", **totals})
    flagged_text = ", ".join(_file_label(f) for f in flagged_lines) or "없음"
    return [
        f"### 영점 h 비현실 플래그 (`{FLAG_ZERO_H}`, v1.7 — 깊이·지표 불변)",
        "",
        "- 정의: 영점 `air_surface_late`이고 h_est > h_max(`DepthConfig.h_max_m` = "
        f"{DEFAULT_DEPTH_CONFIG.h_max_m:g} m)인 측선. 깊이·제외·지표 불변, "
        "활동 안 사본 제외",
        "",
        *_md_table(rows, ZERO_FLAG_TABLE_COLUMNS),
        "",
        f"- 플래그 측선: {flagged_text}",
        "",
    ]


def _multi_event_counts(
    candidates: Sequence[CandidateResult], gt_activities: frozenset[str]
) -> tuple[dict[int, Counter[str]], list[LineFile]]:
    """Per-site counts of the two-event zero table and its lines (spec 7-1 v1.8).

    A line counts when its zero rule is ``air_first_event`` (spec_depth v2.1
    section 4-2-1 step 3). Counted like :func:`_zero_flag_counts`: only lines
    with candidates; ``main_input`` = ``ok`` and kept by the main QC rule.
    """
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    multi_event_lines = []
    for line_file, members in _line_groups(candidates).items():
        site_counts = counts[site_of_activity(line_file[0])]
        site_counts["lines"] += 1
        if ZeroRule.AIR_FIRST_EVENT not in {m.result.zero_rule for m in members}:
            continue
        multi_event_lines.append(line_file)
        ok = [m for m in members if m.result.status == "ok"]
        site_counts["multi_event"] += 1
        site_counts["candidates"] += len(members)
        site_counts["ok"] += len(ok)
        site_counts["main_input"] += sum(not _is_qc_excluded(m, "main") for m in ok)
        site_counts["multi_event_gt_lines"] += line_file[0] in gt_activities
    return counts, sorted(multi_event_lines, key=_line_sort_key)


def _multi_event_lines(
    candidates: Sequence[CandidateResult], evaluation: Evaluation
) -> list[str]:
    """md two-event zero subsection after the zero-flag one (spec 8 (iii) v1.8)."""
    counts, multi_event_lines = _multi_event_counts(
        candidates, evaluation.gt_activities
    )
    count_columns = MULTI_EVENT_TABLE_COLUMNS[1:]
    rows: list[dict[str, object]] = [
        {"site": site, **{c: site_counts[c] for c in count_columns}}
        for site, site_counts in sorted(counts.items())
        if site_counts["multi_event"] > 0
    ]
    totals = {c: sum(int(row[c]) for row in rows) for c in count_columns}
    rows.append({"site": "합계", **totals})
    lines_text = ", ".join(_file_label(f) for f in multi_event_lines) or "없음"
    return [
        f"### 두 사건 영점 (`{ZeroRule.AIR_FIRST_EVENT}`, v1.8 — 깊이 규칙 "
        "spec_depth §4-2-1)",
        "",
        "- 정의: 영점 사건이 2개 이상이라 첫 사건(합쳐진 띠)을 영점으로 쓴 측선"
        f"(`{ZeroRule.AIR_FIRST_EVENT}`, 플래그 `{AIR_MULTI_EVENT}`). 늦은 사건은 "
        "지표 아래 사건으로 본다. 활동 안 사본 제외",
        "",
        *_md_table(rows, MULTI_EVENT_TABLE_COLUMNS),
        "",
        f"- 두 사건 측선: {lines_text}",
        "",
    ]


def write_markdown(
    path: Path,
    evaluation: Evaluation,
    candidates: Sequence[CandidateResult],
    args: argparse.Namespace,
    elapsed_s: float,
    duplicates: DuplicateReport | None = None,
) -> None:
    """Write ``<prefix>.md`` (spec 8).

    Args:
        candidates: The evaluated candidates (v1.7: within-activity copies
            already dropped), so every md count is over them.
        duplicates: v1.7 duplicate facts; ``None`` (tests) leaves item 1
            unchanged and writes no duplicate section.
    """
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
        f"{elapsed_s:.0f} s, 단일 프로세스, "
        f"MKL_CBWR {os.environ.get('MKL_CBWR', '(unset)')}",
        "- 설정: `DepthConfig()` 기본값(4TU 튜닝 없음), "
        f"{_clustering_setting(args)}, 지지 측선 >= {args.min_support_lines}, "
        f"`default_rng({args.baseline_seed})`, 기준선 시행: 주 결과 조합 활동 범위 "
        f"{args.baseline_trials}회, 보조 조합({', '.join(variants_for(args)[1:])})과 "
        f"전역 범위 {args.baseline_trials_aux}회(현장별 값은 같은 시행에서 집계)",
        _qc_setting(args),
        "- 정답 집합: 주 결과 `exclude_x`(code X 제외), 보조 `include_x` "
        "(같은 실행의 매칭 단계에서만 다름, summary·matches의 `gt_set` 열)",
        _candidate_count_line(candidates, evaluation, duplicates),
        *_duplicate_list_lines(duplicates),
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
        *_gate_section_lines(evaluation, args),
        *_auxiliary_v19_section_lines(evaluation, candidates, args),
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
        *_duplicate_section_lines(duplicates, evaluation),
        "## 현장 9 (두 사건 측선)",
        "",
        f"- h_est 분위: {h_quantiles}",
        f"- meta_eps_suspect: {suspects}/{len(site9)}",
        "",
        *_zero_flag_lines(candidates, evaluation),
        *_multi_event_lines(candidates, evaluation),
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
        **_duplicate_dry_run_counts(candidates, lines),
    }
    print(json.dumps(report, ensure_ascii=True))
    return 0


def _duplicate_dry_run_counts(
    candidates: pd.DataFrame, lines: pd.DataFrame
) -> dict[str, int]:
    """dry-run keys of spec 8 v1.7: list checks only, no file is hashed."""
    table = load_duplicate_table(DEFAULT_DUPLICATES)
    copies = table.within_copies()
    run_files = set(_run_line_files(lines))
    candidate_files = zip(
        candidates["activity_id"].astype(str),
        candidates["path"].astype(str),
        strict=True,
    )
    return {
        "duplicate_within_copies": len(copies & run_files),
        "duplicate_pairs_across": sum(
            p.scope == DUP_ACROSS and set(p.files) <= run_files for p in table.pairs
        ),
        "lines_unique": sum(unique_line_counts(lines, table).values()),
        "candidates_on_within_copies": sum(f in copies for f in candidate_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    started = time.perf_counter()
    root, candidates_table, lines_table = _read_inputs(args)
    gt_sets = load_ground_truth_sets(root)
    table = load_duplicate_table(DEFAULT_DUPLICATES)
    # Before any depth: a list that no longer matches the data stops the run.
    hashed_files, hash_seconds = verify_duplicate_hashes(root, lines_table, table)
    line_info = {
        (r.activity_id, r.path): r._asdict()
        for r in lines_table.itertuples(index=False)
    }
    candidates = _run_all_lines(root, candidates_table, args)

    # Copies are still in the csv; evaluation, md and diagnostics drop them.
    eval_candidates = drop_within_copies(candidates, table)
    lines_per_activity = unique_line_counts(lines_table, table)
    evaluation = evaluate_combinations(
        eval_candidates, gt_sets, lines_per_activity, args
    )
    marks = table.marks()
    evaluation.candidate_rows = [candidate_row(s, line_info, marks) for s in candidates]
    evaluation.site11_rows, evaluation.site11_summaries = site11_diagnostic(
        eval_candidates, gt_sets, lines_per_activity, args
    )
    evaluation.multi_trench_activities = sorted(
        a
        for a, items in gt_sets["exclude_x"].items()
        if a in lines_per_activity and any(i.multi_trench for i in items)
    )
    sensitivity_rows = duplicate_sensitivity(
        eval_candidates,
        gt_sets,
        lines_per_activity,
        table,
        args,
        evaluation.summary_rows,
    )
    duplicates = DuplicateReport(
        table=table,
        list_label=_project_relative(DEFAULT_DUPLICATES),
        decisions=keep_one_plan(table, lines_per_activity),
        sensitivity_rows=tuple(sensitivity_rows),
        copy_lines=tuple(
            sorted(table.within_copies() & set(line_info), key=_line_sort_key)
        ),
        n_copy_candidates=len(candidates) - len(eval_candidates),
        gt_counts={a: len(items) for a, items in gt_sets["exclude_x"].items()},
        hashed_files=hashed_files,
        hash_seconds=hash_seconds,
    )
    elapsed_s = time.perf_counter() - started
    _write_outputs(
        evaluation, sensitivity_rows, eval_candidates, args, elapsed_s, duplicates
    )
    LOGGER.info("done in %.0f s", elapsed_s)
    return 0


def _run_all_lines(
    root: Path, candidates_table: pd.DataFrame, args: argparse.Namespace
) -> list[CandidateResult]:
    """Depth (and QC) of every candidate, one line at a time (4-2)."""
    candidates: list[CandidateResult] = []
    for (activity, path), group in candidates_table.groupby(
        ["activity_id", "path"], sort=True
    ):
        radargram = load_fourtu_radargram(root, activity, path, T0Choice.BOTH)
        candidates.extend(run_line(radargram, group, apex_qc=apex_qc_enabled(args)))
        LOGGER.info("%s/%s: %d candidates", activity, path, len(group))
    return candidates


def _project_relative(path: Path) -> str:
    """``path`` relative to the project in POSIX form, else as given."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return path.as_posix()


def _write_outputs(
    evaluation: Evaluation,
    sensitivity_rows: Sequence[Mapping[str, object]],
    eval_candidates: Sequence[CandidateResult],
    args: argparse.Namespace,
    elapsed_s: float,
    duplicates: DuplicateReport,
) -> None:
    """Every csv and the md of spec 8 under ``--out-dir`` with ``--prefix``."""
    out, prefix = args.out_dir, args.prefix
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{prefix}_candidates.csv", evaluation.candidate_rows)
    write_csv(out / f"{prefix}_clusters.csv", evaluation.cluster_rows)
    write_csv(out / f"{prefix}_matches.csv", evaluation.match_rows)
    write_csv(out / f"{prefix}_site11.csv", evaluation.site11_rows)
    write_csv(out / f"{prefix}_summary.csv", evaluation.summary_rows)
    write_csv(out / f"{prefix}_dup_sensitivity.csv", sensitivity_rows)
    write_csv_with_header(
        out / f"{prefix}_ties.csv", evaluation.tie_rows, TIE_CSV_COLUMNS
    )
    write_csv_with_header(
        out / f"{prefix}_unobservable.csv",
        evaluation.unobservable_rows,
        UNOBSERVABLE_CSV_COLUMNS,
    )
    write_markdown(
        out / f"{prefix}.md",
        evaluation,
        eval_candidates,
        args,
        elapsed_s,
        duplicates=duplicates,
    )


if __name__ == "__main__":
    sys.exit(main())
