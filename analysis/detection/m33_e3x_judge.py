"""M3-3 pre-close item 3 (E3x) judgment tool: units -> estimates -> CI -> decision.

Spec: ``docs/processing/m33_e3x_plan.md`` v0.3 (seal 1, 72b0140) §3 and
appendix A-4 (v0.5: the input paths and identity below), with the statistics review
``docs/processing/m33_e3x_stats_review.md`` S1-S6 and its additional review
(boundary rules (1)-(5), check 1 (d)).

Pipeline (plan §3-2, §3-3)

1. Unit values ``y_u``: fold mean (folds 0-4, all present) of the hit
   (hyperbola) or the coverage ratio (band) of the S0 records.
2. Position values ``D_p`` and floor values ``(A_p + B_p) / 2`` per
   hypothesis; ``M*``, ``M*_e``, strata, floor ``F`` and
   ``single_background`` are fixed once on the whole data.
3. Stratified cluster bootstrap (4TU activity / IDS zone clusters per
   background), ``B`` = 20,000, one ``default_rng(BOOT_SEED)``. Draw order:
   HH1, negative control, HH2, HH3, HB1, HB1b, HB2, then the report-only
   items; inside a hypothesis one ``(B, G_g)`` matrix per estimator
   background in the order 4TU, IDS (drawn whenever the background has a
   target cluster, also when the hypothesis is then not evaluable, so the
   stream depends on the data only).
4. Judgment intervals: Satterthwaite t (linear hypotheses and the negative
   control) at the exact level ``1 - 0.05/6``; HH2 the basic (reverse
   percentile) interval. Percentile intervals are report only (R-g).
5. Decision table, floor rule, negative control (a)-(d), HB1 contrast flag.

Inputs asserted (exit 2 otherwise): the injected ``summary.json`` is ``ok``
and judgeable; its record files have the recorded sha; the seal-2 record
matches the current code blobs (``e3x.code_paths``, the ``src`` files
included, per file) and plan tables. The un-injected 0.001 records are the
R-d baseline only when that run's ``summary.json`` is ``ok`` and judgeable
and its ``outputs_sha256`` (which must list ``uninjected001_records.csv``)
matches the files; otherwise ``report_only.R-d.rd_baseline`` is ``null``
and ``report_only.R-d.rd_baseline_reason`` names the first failed check
(:class:`BaselineSkip`). This never refuses the judgment (plan §5 failure
table, uninjected001 row). The record files of a run that is not ``ok`` or
not judgeable are not opened.

``identity``: ``started_at`` is taken at :func:`main` entry and
``finished_at`` when the document is complete; it carries the judge's own
``code_blobs``, the injected and un-injected ``run_id``, ``summary.json``
sha256, and the un-injected ``status`` and ``judgeable``.

v0.5 (plan §5-1, §5-4, A-4): the inputs are the retry folders
``runs/eval/m33/e3x_retry/injected`` and ``e3x_retry/uninjected001`` and the
output is ``e3x_retry/judgment`` (``e3x_retry<k>`` for a later try).
Refused before anything is read (exit 2): an injected, un-injected or
output folder under the first-try root ``runs/eval/m33/e3x`` (the first
try's un-injected output is never the R-d baseline, v5-3). The injected run and
the un-injected run (when its ``summary.json`` exists) must both carry the
given seal record's sha (``run_identity.seal2.seal_record_sha256_lf``) and
its code blobs (v5-3).

Outputs (``runs/eval/m33/e3x_retry/judgment/``, refused if it exists):
``judgment.json`` and ``units_<ID>.csv`` per hypothesis. Stdout carries no
estimate; its last line is ``{"status", "out_dir", "n_hypotheses",
"elapsed_s"}``.

Exit codes: 0 ``ok``; 2 ``input_error`` (input refused, injected not
judgeable, output exists, a first-try folder, runs of another identity);
3 ``error``.
"""

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common
import m32_recipe_inputs as inputs
import m33_e3x as e3x
import m33_e3x_place as place
import m33_inputs

SCHEMA = "m33_e3x_judge/1"
CI_LEVEL = 1 - 0.05 / 6
"""Two-sided level of every judged interval (Bonferroni over m = 6, exact)."""
TAIL = 0.05 / 12
"""One tail of the judged intervals (``(1 - CI_LEVEL) / 2`` as a fraction)."""
REPORT_CI_LEVEL = 0.95
REPORT_TAIL = 0.05 / 2
DELTA = 0.05
FLOOR = 0.10
NEG_FAIL = 0.05
NEG_FLAG = 0.02
NEG_MIN_POSITIONS = 10
NEG_MIN_CLUSTERS = 5
SHIFT_LIMIT_P = place.SHIFT_LIMIT_P
M_STAR_MIN = 0.10
"""``M*``: model mean of ``(Y_on + Y_off) / 2`` at least this (plan §3-2)."""
HH2_THRESHOLD = 0.5
HH2_MIN_POSITIONS_PER_MODEL = 4
SINGLE_BACKGROUND_MIN = 15
CONTRAST_FLAG_DB = 3.0
BOOT_SEED = 20261121
N_BOOT = 20_000
FOLDS = m33_inputs.EXPECTED_FOLDS
BACKGROUNDS = (inputs.DATASET_4TU, inputs.DATASET_IDS)
JUDGED_SNR = place.JUDGED_SNR_DB
MIN_COUNTS = {
    "HH1": (30, 15),
    "HH2": (None, 20),
    "HH3": (40, 20),
    "HB1": (30, 20),
    "HB1b": (20, 12),
    "HB2": (30, 20),
}
"""``(positions, clusters)`` minima of plan §3-4 (HH2: positions per model)."""
HH1_MIN_MODELS = 5
HH2_MIN_MODELS = 12
HH1_OFF_REJECTION_MAX = 0.20
HH3_REJECTION_MAX = 0.30
HYPOTHESES = ("HH1", "HH2", "HH3", "HB1", "HB1b", "HB2")
DEFAULT_INJECTED = e3x.RETRY_OUT_ROOT / "injected"
DEFAULT_UNINJECTED = e3x.RETRY_OUT_ROOT / "uninjected001"
DEFAULT_OUT = e3x.RETRY_OUT_ROOT / "judgment"
EXIT_CODES = {"ok": 0, "input_error": 2, "error": 3}

logger = logging.getLogger("m33_e3x_judge")


class Weighting(StrEnum):
    """Equal weight per model (HH) or per background (HB), plan §3-2."""

    MODEL = "model"
    BACKGROUND = "background"


class Decision(StrEnum):
    """Decision labels of plan §3-3."""

    NOT_EVALUABLE = "not_evaluable"
    CONFIRMED = "confirmed"
    CONFIRMED_POSITIVE = "confirmed_positive"
    CONFIRMED_NEGATIVE = "confirmed_negative"
    REVERSED = "reversed"
    NEGLIGIBLE = "negligible"
    INCONCLUSIVE = "inconclusive"


class CheckOutcome(StrEnum):
    """Negative-control check outcomes (plan §3-3 check 1 (a)-(d))."""

    FAIL = "fail"
    FLAG = "flag"
    PASS = "pass"
    NOT_RUN = "not_run"


WORDING = {
    Decision.NOT_EVALUABLE: "추정 유지(판정 불가: {reason})",
    Decision.CONFIRMED: "확인",
    Decision.CONFIRMED_POSITIVE: "확인(양의 방향)",
    Decision.CONFIRMED_NEGATIVE: "확인(음의 방향)",
    Decision.REVERSED: "기각(반대 방향)",
    Decision.NEGLIGIBLE: "기각(실질 효과 없음)",
    Decision.INCONCLUSIVE: "추정 유지(구간이 넓음)",
}
WORDING_FLOOR = "추정 유지(재현율이 바닥이라 효과 없음을 말할 수 없음)"
WORDING_DEGENERATE = "추정 유지(구간 폭 0: degenerate_ci)"
WORDING_SMALL = " 효과는 있으나 5 %p 미만"
WORDING_NEG_FLAG = " (음성 대조에서도 작은 효과: σ 아래 끝 처리도 영향 가능)"  # noqa: RUF001
WORDING_NEG_NOT_RUN = " (음성 대조 수 부족)"
WORDING_CONTRAST = " (r0 대비 차 포함)"


# ================================================================ bootstrap


@dataclass(frozen=True)
class PositionValue:
    """One target position of a linear hypothesis.

    Attributes:
        difference: ``D_p``.
        floor_value: ``(A_p + B_p) / 2`` (the two compared arms).
    """

    position_id: str
    background: str
    cluster: str
    model: str
    difference: float
    floor_value: float


class Bootstrap:
    """One generator for every draw of a judgment (fixed order, §3-3)."""

    def __init__(self, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_boot = n_boot

    def counts(self, n_clusters: int) -> np.ndarray:
        """``(B, G)`` times each cluster was drawn (``G`` draws with replacement)."""
        draws = self.rng.integers(0, n_clusters, size=(self.n_boot, n_clusters))
        counts = np.zeros((self.n_boot, n_clusters))
        np.add.at(counts, (np.arange(self.n_boot)[:, None], draws), 1.0)
        return counts


@dataclass(frozen=True)
class ClusterDesign:
    """Clusters of one hypothesis per estimator background (sorted ids)."""

    backgrounds: tuple[str, ...]
    clusters: Mapping[str, tuple[str, ...]]
    index: tuple[tuple[str, int], ...]

    @classmethod
    def of(cls, rows: Sequence, backgrounds: Sequence[str]) -> "ClusterDesign":
        """Design from rows with ``background`` and ``cluster`` attributes."""
        present = tuple(b for b in backgrounds if any(r.background == b for r in rows))
        clusters = {
            b: tuple(sorted({r.cluster for r in rows if r.background == b}))
            for b in present
        }
        index = tuple(
            (r.background, clusters[r.background].index(r.cluster)) for r in rows
        )
        return cls(present, clusters, index)

    def n_clusters(self, background: str) -> int:
        return len(self.clusters.get(background, ()))

    def cluster_sizes(self, background: str) -> list[int]:
        sizes = Counter(i for b, i in self.index if b == background)
        return [sizes[i] for i in range(self.n_clusters(background))]


def kish_clusters(sizes: Sequence[int]) -> float:
    """``G_eff = (sum n_c)^2 / sum n_c^2`` (Kish effective cluster count)."""
    return sum(sizes) ** 2 / sum(n * n for n in sizes)


@dataclass(frozen=True)
class Replicates:
    """Weights of every replicate: full and one partial set per background."""

    full: np.ndarray
    partial: Mapping[str, np.ndarray]


def draw_replicates(design: ClusterDesign, bootstrap: Bootstrap) -> Replicates:
    """Draw ``(B, G_g)`` per background (4TU then IDS) and expand to positions.

    A partial replicate resamples one background and keeps every other
    position once (weight 1); it reuses the full replicate's draw (S1).
    """
    n_positions = len(design.index)
    columns = {
        b: [p for p, (bg, _) in enumerate(design.index) if bg == b]
        for b in design.backgrounds
    }
    full = np.ones((bootstrap.n_boot, n_positions))
    partial = {}
    for background in design.backgrounds:
        counts = bootstrap.counts(design.n_clusters(background))
        cluster_of = [design.index[p][1] for p in columns[background]]
        weights = counts[:, cluster_of]
        full[:, columns[background]] = weights
        one = np.ones((bootstrap.n_boot, n_positions))
        one[:, columns[background]] = weights
        partial[background] = one
    return Replicates(full, partial)


def equal_weight_mean(
    weights: np.ndarray, values: np.ndarray, groups: Sequence[str]
) -> np.ndarray:
    """Mean over groups of the weighted group means, per replicate row.

    A group with zero total weight in a replicate leaves that replicate's
    mean, which is divided among the groups left (plan §3-3, [STV] S1).

    Each group mean is ``ref + sum w (x - ref) / sum w`` with ``ref`` the
    group's first value: equal values then give exactly the same mean in
    every replicate, so a constant ``D_p`` has a variance of exactly 0 and
    the ``degenerate_ci`` rule can apply (a plain weighted mean leaves
    rounding noise of about 1e-17 there).
    """
    weights = np.atleast_2d(weights)
    means = []
    for group in sorted(set(groups)):
        mask = np.array([g == group for g in groups])
        reference = values[mask][0]
        total = weights[:, mask].sum(axis=1)
        weighted = weights[:, mask] @ (values[mask] - reference)
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(np.where(total > 0, reference + weighted / total, np.nan))
    stacked = np.vstack(means)
    present = ~np.isnan(stacked)
    return np.where(
        present.any(axis=0),
        np.nansum(stacked, axis=0) / np.maximum(present.sum(axis=0), 1),
        np.nan,
    )


# ============================================================== intervals


@dataclass(frozen=True)
class Interval:
    """A judged (or report) interval and its Satterthwaite pieces."""

    estimate: float
    low: float
    high: float
    method: str
    level: float
    nu: float | None
    v_by_background: Mapping[str, float]
    nu_by_background: Mapping[str, float]
    g_by_background: Mapping[str, int]
    g_eff_by_background: Mapping[str, float]
    percentile_low: float
    percentile_high: float
    degenerate: bool = False
    zero_variance_backgrounds: tuple[str, ...] = ()


def replicate_variance(values: np.ndarray) -> float:
    """Sample variance (``B - 1``); exactly 0 when every replicate is equal.

    ``numpy.var`` of identical values is about 1e-33 (its mean is rounded),
    which would keep ``degenerate_ci`` from ever applying.
    """
    if np.ptp(values) == 0:
        return 0.0
    return float(np.var(values, ddof=1))


def satterthwaite(
    estimate: float,
    partial_values: Mapping[str, np.ndarray],
    n_clusters: Mapping[str, int],
    g_eff: Mapping[str, float],
    tail: float,
) -> tuple[float, float, float | None, dict, dict, bool, tuple[str, ...]]:
    """Satterthwaite t interval from per-background partial replicates (§3-3).

    ``v_g = G_g / (G_g - 1) Var_b``, ``nu_g = max(G_eff - 1, 1)``,
    ``nu = (sum v)^2 / sum_{v_g > 0} v_g^2 / nu_g``; a background with
    ``v_g = 0`` leaves the ``nu`` denominator (``zero_variance_background``);
    ``sum v = 0`` gives ``[estimate, estimate]`` (``degenerate_ci``).

    Returns:
        ``(low, high, nu, v_by_background, nu_by_background, degenerate,
        zero_variance_backgrounds)``.
    """
    v_g, nu_g = {}, {}
    for background, values in partial_values.items():
        g = n_clusters[background]
        v_g[background] = g / (g - 1) * replicate_variance(values)
        nu_g[background] = max(g_eff[background] - 1.0, 1.0)
    total = sum(v_g.values())
    zero = tuple(b for b, v in v_g.items() if v == 0)
    if total == 0:
        return estimate, estimate, None, v_g, nu_g, True, zero
    denominator = sum(v * v / nu_g[b] for b, v in v_g.items() if v > 0)
    nu = total * total / denominator
    half = float(stats.t.ppf(1 - tail, nu)) * math.sqrt(total)
    return estimate - half, estimate + half, nu, v_g, nu_g, False, zero


@dataclass(frozen=True)
class LinearResult:
    """A linear hypothesis (or the control) on its target positions."""

    interval: Interval | None
    not_evaluable_reason: str | None
    n_positions: int
    n_clusters: int
    models: tuple[str, ...]
    by_background: Mapping[str, float]
    floor_mean: float | None
    full_replicates: np.ndarray | None = field(default=None, repr=False)


def _groups(rows: Sequence[PositionValue], weighting: Weighting) -> list[str]:
    if weighting is Weighting.MODEL:
        return [r.model for r in rows]
    return [r.background for r in rows]


def linear_result(
    rows: Sequence[PositionValue],
    weighting: Weighting,
    bootstrap: Bootstrap,
    tail: float = TAIL,
    level: float = CI_LEVEL,
    backgrounds: Sequence[str] = BACKGROUNDS,
) -> LinearResult:
    """Estimate, replicates and interval of one linear hypothesis.

    Draws for every estimator background with a cluster (stream rule of
    the module docstring); ``G_g < 2`` makes it ``single_cluster_background``.
    """
    rows = list(rows)
    if not rows:
        return LinearResult(None, "no_position", 0, 0, (), {}, None)
    design = ClusterDesign.of(rows, backgrounds)
    replicates = draw_replicates(design, bootstrap)
    groups = _groups(rows, weighting)
    values = np.array([r.difference for r in rows], dtype=np.float64)
    floors = np.array([r.floor_value for r in rows], dtype=np.float64)
    ones = np.ones((1, len(rows)))
    estimate = float(equal_weight_mean(ones, values, groups)[0])
    floor_mean = float(equal_weight_mean(ones, floors, groups)[0])
    by_background = {
        b: float(np.mean([r.difference for r in rows if r.background == b]))
        for b in design.backgrounds
    }
    n_clusters = {b: design.n_clusters(b) for b in design.backgrounds}
    total_clusters = sum(n_clusters.values())
    models = tuple(sorted({r.model for r in rows}))
    if any(g < 2 for g in n_clusters.values()):
        return LinearResult(
            None,
            "single_cluster_background",
            len(rows),
            total_clusters,
            models,
            by_background,
            floor_mean,
        )
    full = equal_weight_mean(replicates.full, values, groups)
    partial = {
        b: equal_weight_mean(w, values, groups) for b, w in replicates.partial.items()
    }
    g_eff = {b: kish_clusters(design.cluster_sizes(b)) for b in design.backgrounds}
    low, high, nu, v_g, nu_g, degenerate, zero = satterthwaite(
        estimate, partial, n_clusters, g_eff, tail
    )
    percentile = np.quantile(full, [tail, 1 - tail])
    interval = Interval(
        estimate=estimate,
        low=low,
        high=high,
        method="satterthwaite_t",
        level=level,
        nu=nu,
        v_by_background=v_g,
        nu_by_background=nu_g,
        g_by_background=n_clusters,
        g_eff_by_background=g_eff,
        percentile_low=float(percentile[0]),
        percentile_high=float(percentile[1]),
        degenerate=degenerate,
        zero_variance_backgrounds=zero,
    )
    return LinearResult(
        interval,
        None,
        len(rows),
        total_clusters,
        models,
        by_background,
        floor_mean,
        full,
    )


# =============================================================== decisions


@dataclass(frozen=True)
class Verdict:
    """Decision, its reason and wording of one hypothesis."""

    decision: Decision
    reason: str | None
    wording: str
    small_effect: bool = False


def directional_verdict(
    interval: Interval | None,
    floor_mean: float | None,
    not_evaluable_reason: str | None,
    two_sided: bool = False,
) -> Verdict:
    """Plan §3-3 decision table (HH3, HB1, HB1b, HB2; HH1 when ``two_sided``).

    Order: not evaluable; ``degenerate_ci`` with a non-zero estimate;
    lower bound > 0; upper bound < 0; inside the open band ``(-d, +d)`` with
    ``F >= 0.10`` (``negligible``) or below it (``floor``); else
    ``inconclusive``. Boundaries: a bound exactly 0 confirms nothing, a
    bound exactly +-d is not inside, ``F`` exactly 0.10 meets the floor.
    """
    if not_evaluable_reason is not None or interval is None:
        reason = not_evaluable_reason or "no_interval"
        return Verdict(
            Decision.NOT_EVALUABLE,
            reason,
            WORDING[Decision.NOT_EVALUABLE].format(reason=reason),
        )
    if interval.degenerate and interval.estimate != 0:
        return Verdict(Decision.INCONCLUSIVE, "degenerate_ci", WORDING_DEGENERATE)
    if interval.low > 0:
        label = Decision.CONFIRMED_POSITIVE if two_sided else Decision.CONFIRMED
        small = interval.estimate < DELTA
        wording = WORDING[label] + (WORDING_SMALL if small else "")
        return Verdict(label, None, wording, small)
    if interval.high < 0:
        label = Decision.CONFIRMED_NEGATIVE if two_sided else Decision.REVERSED
        return Verdict(label, None, WORDING[label])
    if interval.low > -DELTA and interval.high < DELTA:
        if floor_mean is not None and floor_mean >= FLOOR:
            return Verdict(Decision.NEGLIGIBLE, None, WORDING[Decision.NEGLIGIBLE])
        return Verdict(Decision.INCONCLUSIVE, "floor", WORDING_FLOOR)
    return Verdict(Decision.INCONCLUSIVE, None, WORDING[Decision.INCONCLUSIVE])


def eta_verdict(low: float | None, high: float | None, reason: str | None) -> Verdict:
    """HH2: lower > 0.5 confirmed, upper < 0.5 reversed, else inconclusive."""
    if reason is not None or low is None or high is None:
        reason = reason or "no_interval"
        return Verdict(
            Decision.NOT_EVALUABLE,
            reason,
            WORDING[Decision.NOT_EVALUABLE].format(reason=reason),
        )
    if low > HH2_THRESHOLD:
        return Verdict(Decision.CONFIRMED, None, WORDING[Decision.CONFIRMED])
    if high < HH2_THRESHOLD:
        return Verdict(Decision.REVERSED, None, WORDING[Decision.REVERSED])
    return Verdict(Decision.INCONCLUSIVE, None, WORDING[Decision.INCONCLUSIVE])


def negative_control_outcome(result: LinearResult) -> CheckOutcome:
    """Check 1 (a)-(d): ``not_run`` below 10 positions or 5 clusters or with a
    single-cluster background; else the interval excluding 0 with
    ``|d| >= 0.05`` fails, ``0.02 <= |d| < 0.05`` flags, anything else passes
    (``|d|`` = 0.05 fails and 0.02 flags: both boundaries included).
    """
    if (
        result.n_positions < NEG_MIN_POSITIONS
        or result.n_clusters < NEG_MIN_CLUSTERS
        or result.not_evaluable_reason is not None
        or result.interval is None
    ):
        return CheckOutcome.NOT_RUN
    interval = result.interval
    excludes_zero = interval.low > 0 or interval.high < 0
    size = abs(interval.estimate)
    if excludes_zero and size >= NEG_FAIL:
        return CheckOutcome.FAIL
    if excludes_zero and size >= NEG_FLAG:
        return CheckOutcome.FLAG
    return CheckOutcome.PASS


def basic_interval(
    estimate: float, replicates: np.ndarray, tail: float = TAIL
) -> tuple[float, float]:
    """``[2 t - q_{1-a}, 2 t - q_a]`` (``numpy.quantile`` linear, not clipped)."""
    low_q, high_q = np.quantile(replicates, [tail, 1 - tail])
    return 2 * estimate - float(high_q), 2 * estimate - float(low_q)


def percentile_interval(
    replicates: np.ndarray, tail: float = TAIL
) -> tuple[float, float]:
    """Report-only percentile interval of the same replicates."""
    low_q, high_q = np.quantile(replicates, [tail, 1 - tail])
    return float(low_q), float(high_q)


# =================================================================== HH2


@dataclass(frozen=True)
class EtaRow:
    """One HH2 position: ``Y_on`` over ``S_J``."""

    position_id: str
    background: str
    cluster: str
    model: str
    value: float


@dataclass(frozen=True)
class Decomposition:
    """Type II variance split of an additive background + model model."""

    eta2_p: float
    omega2_p: float
    ss_model: float
    ss_res: float
    df_model: int
    df_res: int
    n: float


def _weighted_rss(
    design: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, int]:
    root = np.sqrt(w)
    a, b = design * root[:, None], y * root
    coef, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    residual = b - a @ coef
    return float(residual @ residual), int(rank)


def decompose(
    backgrounds: Sequence[str],
    models: Sequence[str],
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> Decomposition:
    """``SS_model`` after background (Type II) and the residual of the full model.

    Integer weights stand for repeated positions (a bootstrap replicate);
    a model without weight drops out and the degrees of freedom are those of
    the rank of the design that is left.
    """
    w = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    keep = w > 0
    bgs = [b for b, k in zip(backgrounds, keep, strict=True) if k]
    mds = [m for m, k in zip(models, keep, strict=True) if k]
    y, w = np.asarray(values, float)[keep], w[keep]
    intercept = np.ones((len(y), 1))
    bg_cols = [
        np.array([b == level for b in bgs], float) for level in sorted(set(bgs))[1:]
    ]
    md_cols = [
        np.array([m == level for m in mds], float) for level in sorted(set(mds))[1:]
    ]
    reduced = np.column_stack([intercept, *bg_cols]) if bg_cols else intercept
    full = np.column_stack([reduced, *md_cols]) if md_cols else reduced
    rss_reduced, rank_reduced = _weighted_rss(reduced, y, w)
    rss_full, rank_full = _weighted_rss(full, y, w)
    n = float(w.sum())
    ss_model = max(rss_reduced - rss_full, 0.0)
    df_model = rank_full - rank_reduced
    df_res = round(n) - rank_full
    total = ss_model + rss_full
    eta = ss_model / total if total > 0 else math.nan
    omega = math.nan
    if df_model > 0 and df_res > 0:
        ms_model, ms_res = ss_model / df_model, rss_full / df_res
        denominator = df_model * ms_model + (n - df_model) * ms_res
        if denominator > 0:
            omega = df_model * (ms_model - ms_res) / denominator
    return Decomposition(eta, omega, ss_model, rss_full, df_model, df_res, n)


@dataclass(frozen=True)
class EtaResult:
    """HH2 estimates and intervals (judged: ``eta2_p`` basic interval)."""

    estimate: float | None
    low: float | None
    high: float | None
    percentile: tuple[float, float] | None
    omega2_p: dict
    eta2_p_mstar: dict
    n_positions: int
    n_clusters: int
    models: tuple[str, ...]
    g_by_background: Mapping[str, int]
    not_evaluable_reason: str | None
    n_nan_replicates: int = 0


def replicate_decompositions(
    rows: Sequence[EtaRow],
    weights: np.ndarray,
    model_filter: Callable[[str], bool] = lambda _: True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(eta2_p, omega2_p)`` of every replicate row of ``weights``."""
    selected = [i for i, r in enumerate(rows) if model_filter(r.model)]
    bgs = [rows[i].background for i in selected]
    mds = [rows[i].model for i in selected]
    values = np.array([rows[i].value for i in selected])
    eta = np.empty(weights.shape[0])
    omega = np.empty(weights.shape[0])
    for b in range(weights.shape[0]):
        result = decompose(bgs, mds, values, weights[b, selected])
        eta[b], omega[b] = result.eta2_p, result.omega2_p
    return eta, omega


def _eta_summary(estimate: float, replicates: np.ndarray, tail: float) -> dict:
    finite = replicates[np.isfinite(replicates)]
    if not math.isfinite(estimate) or finite.size == 0:
        return {"estimate": None, "low": None, "high": None}
    low, high = basic_interval(estimate, finite, tail)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "percentile": percentile_interval(finite, tail),
    }


def eta_result(
    rows: Sequence[EtaRow],
    m_star: Sequence[str],
    bootstrap: Bootstrap,
    tail: float = TAIL,
) -> EtaResult:
    """HH2 on models with at least four positions (structural zeros included)."""
    per_model = Counter(r.model for r in rows)
    kept = [r for r in rows if per_model[r.model] >= HH2_MIN_POSITIONS_PER_MODEL]
    models = tuple(sorted({r.model for r in kept}))
    if not kept:
        return EtaResult(None, None, None, None, {}, {}, 0, 0, (), {}, "no_position")
    design = ClusterDesign.of(kept, BACKGROUNDS)
    replicates = draw_replicates(design, bootstrap)
    n_clusters = {b: design.n_clusters(b) for b in design.backgrounds}
    total_clusters = sum(n_clusters.values())
    reason = None
    if len(models) < HH2_MIN_MODELS:
        reason = "models_below_minimum"
    elif total_clusters < MIN_COUNTS["HH2"][1]:
        reason = "clusters_below_minimum"
    elif any(g < 2 for g in n_clusters.values()):
        reason = "single_cluster_background"
    base = EtaResult(
        None,
        None,
        None,
        None,
        {},
        {},
        len(kept),
        total_clusters,
        models,
        n_clusters,
        reason,
    )
    if reason is not None:
        return base
    point = decompose(
        [r.background for r in kept],
        [r.model for r in kept],
        np.array([r.value for r in kept]),
    )
    everything, omega = replicate_decompositions(kept, replicates.full)
    mstar = set(m_star)
    mstar_rows = [r for r in kept if r.model in mstar]
    mstar_summary: dict = {"estimate": None}
    if mstar_rows and len({r.model for r in mstar_rows}) > 1:
        mstar_point = decompose(
            [r.background for r in mstar_rows],
            [r.model for r in mstar_rows],
            np.array([r.value for r in mstar_rows]),
        )
        mstar_reps, _ = replicate_decompositions(
            kept, replicates.full, lambda model: model in mstar
        )
        mstar_summary = _eta_summary(mstar_point.eta2_p, mstar_reps, tail)
    finite = everything[np.isfinite(everything)]
    judged = _eta_summary(point.eta2_p, everything, tail)
    return replace(
        base,
        estimate=judged["estimate"],
        low=judged["low"],
        high=judged["high"],
        percentile=judged.get("percentile"),
        omega2_p=_eta_summary(point.omega2_p, omega, tail),
        eta2_p_mstar=mstar_summary,
        n_nan_replicates=int(everything.size - finite.size),
    )


# ============================================================ unit tables


def _true(value: object) -> bool:
    return value is True or value == "True"


def fold_means(
    rows: Iterable[Mapping[str, str]],
    key: Callable[[Mapping[str, str]], tuple],
    value: str,
    folds: Sequence[int] = FOLDS,
) -> dict[tuple, float]:
    """``y_u``: fold mean of ``value`` per unit; units missing a fold drop."""
    collected: dict[tuple, dict[int, float]] = {}
    for row in rows:
        if not _true(row["accepted"]) or row[value] in ("", None):
            continue
        raw = row[value]
        number = float(_true(raw)) if raw in ("True", "False") else float(raw)
        collected.setdefault(key(row), {})[int(row["fold"])] = number
    return {
        unit: sum(by_fold[f] for f in folds) / len(folds)
        for unit, by_fold in collected.items()
        if all(f in by_fold for f in folds)
    }


def hyperbola_key(row: Mapping[str, str]) -> tuple:
    """``(position, taper, SNR, polarity)``."""
    return (
        row["position_id"],
        row["taper"],
        float(row["snr_db"]),
        int(row["polarity"]),
    )


def band_key(row: Mapping[str, str]) -> tuple:
    """``(position, row px, length arm, texture, recipe, SNR, polarity)``."""
    return (
        row["position_id"],
        float(row["bin_target_px"]),
        row["length_arm"],
        row["texture"],
        row["recipe"],
        float(row["snr_db"]),
        int(row["polarity"]),
    )


def arm_mean(
    units: Mapping[tuple, float], prefix: tuple, snrs: Iterable[float]
) -> float | None:
    """Mean of ``y_u`` over SNR x polarity; ``None`` when a unit is missing."""
    values = []
    for snr in snrs:
        for polarity in place.inject.POLARITIES:
            unit = (*prefix, float(snr), polarity)
            if unit not in units:
                return None
            values.append(units[unit])
    return sum(values) / len(values)


@dataclass(frozen=True)
class HyperbolaPosition:
    """Fold-free facts of a hyperbola position (from its records)."""

    position_id: str
    background: str
    cluster: str
    model: str
    r_edge_db: float
    stratum: str
    shares_r3_activity: bool


@dataclass(frozen=True)
class BandPosition:
    """Fold-free facts of a band position (from its records and the plan)."""

    position_id: str
    background: str
    cluster: str
    model: str
    shares_r3_activity: bool
    max_shift_p: Mapping[tuple[float, str], float]


def hyperbola_positions(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, HyperbolaPosition]:
    """Positions from the records (first row of each)."""
    found = {}
    for row in rows:
        if row["position_id"] in found:
            continue
        found[row["position_id"]] = HyperbolaPosition(
            row["position_id"],
            row["dataset"],
            row["cluster_id"],
            row["model"],
            float(row["r_edge_db"]),
            row["edge_stratum"],
            _true(row["shares_r3_activity"]),
        )
    return found


def band_positions(rows: Iterable[Mapping[str, str]]) -> dict[str, BandPosition]:
    """Positions from the records, with the shift of each (row, arm)."""
    facts: dict[str, dict] = {}
    for row in rows:
        entry = facts.setdefault(
            row["position_id"],
            {"row": row, "shift": {}},
        )
        key = (float(row["bin_target_px"]), row["length_arm"])
        entry["shift"][key] = float(row["shift_P"])
    return {
        pid: BandPosition(
            pid,
            e["row"]["dataset"],
            e["row"]["cluster_id"],
            e["row"]["model"],
            _true(e["row"]["shares_r3_activity"]),
            e["shift"],
        )
        for pid, e in facts.items()
    }


# ======================================================= hypothesis tables


ON, OFF = e3x.TAPER_ON, e3x.TAPER_OFF


def m_star_models(
    units: Mapping[tuple, float], positions: Mapping[str, HyperbolaPosition]
) -> tuple[str, ...]:
    """``M*``: models whose mean ``(Y_on + Y_off) / 2`` over ``S_J`` >= 0.10."""
    by_model: dict[str, list[float]] = {}
    for pid, position in positions.items():
        on = arm_mean(units, (pid, ON), JUDGED_SNR)
        off = arm_mean(units, (pid, OFF), JUDGED_SNR)
        if on is None or off is None:
            continue
        by_model.setdefault(position.model, []).append((on + off) / 2)
    return tuple(
        sorted(m for m, v in by_model.items() if statistics.fmean(v) >= M_STAR_MIN)
    )


def _position_value(position, difference: float, floor_value: float) -> PositionValue:
    return PositionValue(
        position.position_id,
        position.background,
        position.cluster,
        position.model,
        difference,
        floor_value,
    )


def hh1_rows(
    units: Mapping[tuple, float],
    positions: Mapping[str, HyperbolaPosition],
    m_star: Sequence[str],
    stratum: str = place.EdgeStratum.MANIPULATED.value,
    snr_sets: Callable[[HyperbolaPosition], tuple[float, ...]] | None = None,
) -> list[PositionValue]:
    """HH1 (or control / report) rows: ``D_p`` = on - off over the unit SNRs.

    Args:
        stratum: ``manipulated`` (HH1, ``S_p^edge``), ``negative_control``
            (``S_J``) or ``"all"`` (report: every ``M*`` position, ``S_J``).
        snr_sets: Override of the SNR set per position (report variants).
    """
    rows = []
    for pid, position in sorted(positions.items()):
        if position.model not in m_star:
            continue
        if stratum != "all" and position.stratum != stratum:
            continue
        if snr_sets is not None:
            snrs = snr_sets(position)
        elif stratum == place.EdgeStratum.MANIPULATED.value:
            snrs = place.edge_snr_set(position.r_edge_db)
        else:
            snrs = JUDGED_SNR
        if not snrs:
            continue
        on = arm_mean(units, (pid, ON), snrs)
        off = arm_mean(units, (pid, OFF), snrs)
        if on is None or off is None:
            continue
        rows.append(_position_value(position, on - off, (on + off) / 2))
    return rows


def off_rejection_share(
    records: Iterable[Mapping[str, str]],
    units: Mapping[tuple, float],
    positions: Mapping[str, HyperbolaPosition],
    m_star: Sequence[str],
) -> float | None:
    """Share of manipulated ``M*`` positions whose off arm lost a unit to int16."""
    rejected = {
        row["position_id"]
        for row in records
        if row["taper"] == OFF and row["reject_reason"] == place.OFF_INT16_REJECTION
    }
    eligible = [
        pid
        for pid, p in positions.items()
        if p.model in m_star
        and p.stratum == place.EdgeStratum.MANIPULATED.value
        and arm_mean(units, (pid, ON), place.edge_snr_set(p.r_edge_db)) is not None
    ]
    if not eligible:
        return None
    return sum(pid in rejected for pid in eligible) / len(eligible)


def hh3_rows(
    units: Mapping[tuple, float],
    positions: Mapping[str, HyperbolaPosition],
    m_star: Sequence[str],
) -> list[PositionValue]:
    """HH3 rows: ``D_p`` = 12 dB on - 30 dB on (both polarities)."""
    rows = []
    for pid, position in sorted(positions.items()):
        if position.model not in m_star:
            continue
        low = arm_mean(units, (pid, ON), (12.0,))
        high = arm_mean(units, (pid, ON), (30.0,))
        if low is None or high is None:
            continue
        rows.append(_position_value(position, low - high, (low + high) / 2))
    return rows


def hh3_rejection_share(
    units: Mapping[tuple, float],
    positions: Mapping[str, HyperbolaPosition],
    m_star: Sequence[str],
) -> float | None:
    """Share of ``M*`` positions (12 dB on present) without both 30 dB units."""
    eligible = [
        pid
        for pid, p in positions.items()
        if p.model in m_star and arm_mean(units, (pid, ON), (12.0,)) is not None
    ]
    if not eligible:
        return None
    missing = sum(arm_mean(units, (pid, ON), (30.0,)) is None for pid in eligible)
    return missing / len(eligible)


BAND_FLAT = (place.LONG_ARM, e3x.TEXTURE_NONE, e3x.RECIPE_R0)


def band_arm(
    units: Mapping[tuple, float],
    pid: str,
    row_px: float,
    arm: str = place.LONG_ARM,
    texture: str = e3x.TEXTURE_NONE,
    recipe: str = e3x.RECIPE_R0,
    snrs: Iterable[float] = JUDGED_SNR,
) -> float | None:
    """``Y_{p, row, arm}`` over the SNRs and both polarities."""
    return arm_mean(units, (pid, float(row_px), arm, texture, recipe), snrs)


def _mean_or_none(*values: float | None) -> float | None:
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def hb_rows(
    name: str,
    units: Mapping[tuple, float],
    positions: Mapping[str, BandPosition],
    keep: Callable[[BandPosition], bool] = lambda _: True,
) -> list[PositionValue]:
    """HB1, HB1b or HB2 rows (plan §3-2 table)."""
    rows = []
    for pid, position in sorted(positions.items()):
        if not keep(position):
            continue
        in_range = _mean_or_none(band_arm(units, pid, 47.0), band_arm(units, pid, 68.0))
        if name == "HB1":
            other = band_arm(units, pid, 28.0)
        elif name == "HB1b":
            # Plan §3-2 table: HB1b positions have all four rows (28, 47, 68,
            # 90 px), so 28 px is required although it is not in the difference.
            has_28 = band_arm(units, pid, 28.0) is not None
            other = band_arm(units, pid, 90.0) if has_28 else None
        else:
            other = _mean_or_none(
                band_arm(units, pid, 47.0, place.SHORT_ARM),
                band_arm(units, pid, 68.0, place.SHORT_ARM),
            )
        if in_range is None or other is None:
            continue
        rows.append(_position_value(position, in_range - other, (in_range + other) / 2))
    return rows


def planned_band_counts(plan_dir: Path) -> dict[str, dict[str, int]]:
    """Planned positions per background of HB1, HB1b, HB2 (``single_background``)."""
    positions = inputs.read_csv(plan_dir / "band_positions.csv")
    arms = inputs.read_csv(plan_dir / "band_arms.csv")
    selected = e3x.band_hypothesis_positions(positions, arms)
    return {
        name: dict(Counter(row["dataset"] for row in rows))
        for name, rows in selected.items()
    }


def kept_backgrounds(planned: Mapping[str, int]) -> tuple[tuple[str, ...], bool]:
    """Backgrounds with at least 15 planned positions, and the single flag."""
    kept = tuple(b for b in BACKGROUNDS if planned.get(b, 0) >= SINGLE_BACKGROUND_MIN)
    present = tuple(b for b in BACKGROUNDS if planned.get(b, 0) > 0)
    return kept, len(kept) < len(present)


def contrast_flag(band_records: Iterable[Mapping[str, str]]) -> dict:
    """HB1 contrast check (plan §3-3 2): 28 px median ``c_lat`` 3 dB under both."""
    values: dict[float, dict[str, float]] = {}
    for row in band_records:
        if not _true(row["accepted"]) or row["c_lat"] in ("", None):
            continue
        if (row["length_arm"], row["texture"], row["recipe"]) != BAND_FLAT:
            continue
        if float(row["snr_db"]) not in JUDGED_SNR:
            continue
        # One value per realization and position: c_lat does not depend on the fold.
        unit = (row["realization_id"], row["position_id"])
        values.setdefault(float(row["bin_target_px"]), {})[unit] = float(row["c_lat"])
    medians = {row: statistics.median(v.values()) for row, v in values.items() if v}
    if not all(r in medians for r in (28.0, 47.0, 68.0)):
        return {"medians_db": medians, "flag": False, "evaluated": False}
    flag = all(medians[28.0] < medians[r] - CONTRAST_FLAG_DB for r in (47.0, 68.0))
    return {"medians_db": medians, "flag": flag, "evaluated": True}


# ============================================================ evaluation


def _minimum_reason(name: str, result: LinearResult) -> str | None:
    positions, clusters = MIN_COUNTS[name]
    if positions is not None and result.n_positions < positions:
        return "positions_below_minimum"
    if result.n_clusters < clusters:
        return "clusters_below_minimum"
    return None


def linear_entry(
    name: str,
    result: LinearResult,
    verdict: Verdict,
    extra: Mapping | None = None,
) -> dict:
    """``judgment.json`` block of a linear hypothesis (A-4 keys)."""
    interval = result.interval
    entry = {
        "estimate": None if interval is None else interval.estimate,
        "ci_method": "satterthwaite_t",
        "ci_low": None if interval is None else interval.low,
        "ci_high": None if interval is None else interval.high,
        "ci_level": CI_LEVEL,
        "nu": None if interval is None else interval.nu,
        "v_by_background": None if interval is None else dict(interval.v_by_background),
        "nu_by_background": None
        if interval is None
        else dict(interval.nu_by_background),
        "g_by_background": None if interval is None else dict(interval.g_by_background),
        "g_eff_by_background": (
            None if interval is None else dict(interval.g_eff_by_background)
        ),
        "ci_percentile_low": None if interval is None else interval.percentile_low,
        "ci_percentile_high": None if interval is None else interval.percentile_high,
        "by_background": dict(result.by_background),
        "n_positions": result.n_positions,
        "n_clusters": result.n_clusters,
        "models": list(result.models),
        "floor_mean": result.floor_mean,
        "floor_ok": None if result.floor_mean is None else result.floor_mean >= FLOOR,
        "decision": verdict.decision.value,
        "wording": verdict.wording,
        "small_effect": verdict.small_effect,
        "degenerate_ci": bool(interval and interval.degenerate),
        "zero_variance_background": (
            [] if interval is None else list(interval.zero_variance_backgrounds)
        ),
        "not_evaluable_reason": verdict.reason
        if verdict.decision is Decision.NOT_EVALUABLE
        else None,
        "decision_reason": verdict.reason,
        "single_background": False,
        "single_cluster_background": result.not_evaluable_reason
        == "single_cluster_background",
        "delta": DELTA,
    }
    return entry | dict(extra or {})


@dataclass(frozen=True)
class JudgmentData:
    """Units and positions of one metric choice (S0 judged, S3 report)."""

    hyperbola_units: Mapping[tuple, float]
    band_units: Mapping[tuple, float]
    hyperbolas: Mapping[str, HyperbolaPosition]
    bands: Mapping[str, BandPosition]
    hyperbola_records: Sequence[Mapping[str, str]]
    band_records: Sequence[Mapping[str, str]]
    planned_band: Mapping[str, Mapping[str, int]]


def load_data(
    hyperbola_records: Sequence[Mapping[str, str]],
    band_records: Sequence[Mapping[str, str]],
    planned_band: Mapping[str, Mapping[str, int]],
    hit_column: str = "hit",
    coverage_column: str = "coverage",
) -> JudgmentData:
    """Unit tables of the judged (S0) or the S3 values."""
    return JudgmentData(
        fold_means(hyperbola_records, hyperbola_key, hit_column),
        fold_means(band_records, band_key, coverage_column),
        hyperbola_positions(hyperbola_records),
        band_positions(band_records),
        hyperbola_records,
        band_records,
        planned_band,
    )


def hh1_verdict(
    result: LinearResult,
    control: CheckOutcome,
    off_rejection: float | None,
) -> Verdict:
    """HH1 label: minimum counts, ``|M*_e|``, off int16 share, control (a)-(d).

    There is no fallback estimand (user decision v3-6): too few manipulated
    positions or models make HH1 ``not_evaluable``. Only a failed control
    (a) changes the label; (b) and (d) add a phrase.
    """
    reason = result.not_evaluable_reason or _minimum_reason("HH1", result)
    if reason is None and len(result.models) < HH1_MIN_MODELS:
        reason = "models_below_minimum"
    if (
        reason is None
        and off_rejection is not None
        and off_rejection > HH1_OFF_REJECTION_MAX
    ):
        reason = "off_int16_rejection"
    if reason is None and control is CheckOutcome.FAIL:
        reason = "negative_control_failed"
    verdict = directional_verdict(result.interval, result.floor_mean, reason, True)
    suffix = {
        CheckOutcome.FLAG: WORDING_NEG_FLAG,
        CheckOutcome.NOT_RUN: WORDING_NEG_NOT_RUN,
    }
    return replace(verdict, wording=verdict.wording + suffix.get(control, ""))


def evaluate_hh1(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> tuple[dict, dict]:
    """HH1 and its negative control (drawn in that order)."""
    rows = hh1_rows(data.hyperbola_units, data.hyperbolas, m_star)
    result = linear_result(rows, Weighting.MODEL, bootstrap)
    control_rows = hh1_rows(
        data.hyperbola_units,
        data.hyperbolas,
        m_star,
        place.EdgeStratum.NEGATIVE_CONTROL.value,
    )
    control = linear_result(control_rows, Weighting.MODEL, bootstrap)
    outcome = negative_control_outcome(control)
    share = off_rejection_share(
        data.hyperbola_records, data.hyperbola_units, data.hyperbolas, m_star
    )
    verdict = hh1_verdict(result, outcome, share)
    control_block = {
        "estimate": None if control.interval is None else control.interval.estimate,
        "ci_low": None if control.interval is None else control.interval.low,
        "ci_high": None if control.interval is None else control.interval.high,
        "ci_level": CI_LEVEL,
        "nu": None if control.interval is None else control.interval.nu,
        "n_positions": control.n_positions,
        "n_clusters": control.n_clusters,
        "models": list(control.models),
        "outcome": outcome.value,
        "not_run_reason": control.not_evaluable_reason,
    }
    entry = linear_entry(
        "HH1",
        result,
        verdict,
        {
            "m_star": list(m_star),
            "m_star_e": list(result.models),
            "off_int16_rejection_share": share,
            "manipulation_checks": {"negative_control": control_block},
        },
    )
    return entry, {"rows": rows, "control_rows": control_rows}


def evaluate_hh3(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> tuple[dict, list]:
    rows = hh3_rows(data.hyperbola_units, data.hyperbolas, m_star)
    result = linear_result(rows, Weighting.MODEL, bootstrap)
    reason = result.not_evaluable_reason or _minimum_reason("HH3", result)
    share = hh3_rejection_share(data.hyperbola_units, data.hyperbolas, m_star)
    if reason is None and share is not None and share > HH3_REJECTION_MAX:
        reason = "rejection_30db"
    verdict = directional_verdict(result.interval, result.floor_mean, reason)
    return linear_entry("HH3", result, verdict, {"rejection_30db_share": share}), rows


def evaluate_hb(
    name: str,
    data: JudgmentData,
    bootstrap: Bootstrap,
    contrast: Mapping | None = None,
    keep: Callable[[BandPosition], bool] = lambda _: True,
    tail: float = TAIL,
) -> tuple[dict, list]:
    """HB1, HB1b or HB2 (background-equal weight, ``single_background``)."""
    kept, single = kept_backgrounds(data.planned_band.get(name, {}))
    rows = [
        r
        for r in hb_rows(name, data.band_units, data.bands, keep)
        if r.background in kept
    ]
    result = linear_result(
        rows,
        Weighting.BACKGROUND,
        bootstrap,
        tail,
        CI_LEVEL if tail == TAIL else REPORT_CI_LEVEL,
        kept,
    )
    reason = result.not_evaluable_reason or _minimum_reason(name, result)
    if not kept:
        reason = "no_background_with_15_positions"
    verdict = directional_verdict(result.interval, result.floor_mean, reason)
    if name == "HB1" and contrast and contrast.get("flag"):
        verdict = replace(verdict, wording=verdict.wording + WORDING_CONTRAST)
    extra = {"single_background": single, "backgrounds_used": list(kept)}
    if name == "HB1" and contrast is not None:
        extra["manipulation_checks"] = {"contrast": dict(contrast)}
    return linear_entry(name, result, verdict, extra), rows


def hh2_rows(data: JudgmentData) -> list[EtaRow]:
    """HH2 rows: ``Y_on`` over ``S_J`` of every position with all four units."""
    rows = []
    for pid, position in sorted(data.hyperbolas.items()):
        value = arm_mean(data.hyperbola_units, (pid, ON), JUDGED_SNR)
        if value is not None:
            rows.append(
                EtaRow(
                    pid, position.background, position.cluster, position.model, value
                )
            )
    return rows


def evaluate_hh2(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> tuple[dict, list]:
    rows = hh2_rows(data)
    result = eta_result(rows, m_star, bootstrap)
    verdict = eta_verdict(result.low, result.high, result.not_evaluable_reason)
    entry = {
        "estimate": result.estimate,
        "ci_method": "basic_bootstrap",
        "ci_low": result.low,
        "ci_high": result.high,
        "ci_level": CI_LEVEL,
        "ci_percentile_low": None
        if result.percentile is None
        else result.percentile[0],
        "ci_percentile_high": None
        if result.percentile is None
        else result.percentile[1],
        "threshold": HH2_THRESHOLD,
        "n_positions": result.n_positions,
        "n_clusters": result.n_clusters,
        "models": list(result.models),
        "g_by_background": dict(result.g_by_background),
        "omega2_p": result.omega2_p,
        "eta2_p_mstar": result.eta2_p_mstar,
        "n_nan_replicates": result.n_nan_replicates,
        "decision": verdict.decision.value,
        "wording": verdict.wording,
        "not_evaluable_reason": result.not_evaluable_reason,
    }
    return entry, rows


def judge_hypotheses(data: JudgmentData, bootstrap: Bootstrap) -> tuple[dict, dict]:
    """The six hypotheses and the control in the fixed draw order."""
    m_star = m_star_models(data.hyperbola_units, data.hyperbolas)
    tables: dict[str, list] = {}
    hh1, hh1_tables = evaluate_hh1(data, m_star, bootstrap)
    tables["HH1"], tables["HH1_negative_control"] = (
        hh1_tables["rows"],
        hh1_tables["control_rows"],
    )
    hh2, tables["HH2"] = evaluate_hh2(data, m_star, bootstrap)
    hh3, tables["HH3"] = evaluate_hh3(data, m_star, bootstrap)
    contrast = contrast_flag(data.band_records)
    results = {"HH1": hh1, "HH2": hh2, "HH3": hh3}
    for name in ("HB1", "HB1b", "HB2"):
        results[name], tables[name] = evaluate_hb(
            name, data, bootstrap, contrast if name == "HB1" else None
        )
    return results, tables


# ============================================================ report only


def report_linear(
    rows: Sequence[PositionValue], weighting: Weighting, bootstrap: Bootstrap
) -> dict:
    """95 % Satterthwaite (no multiplicity correction) of a report quantity."""
    result = linear_result(rows, weighting, bootstrap, REPORT_TAIL, REPORT_CI_LEVEL)
    interval = result.interval
    return {
        "estimate": None if interval is None else interval.estimate,
        "ci_low": None if interval is None else interval.low,
        "ci_high": None if interval is None else interval.high,
        "ci_level": REPORT_CI_LEVEL,
        "note": "다중 비교 보정 없음",
        "n_positions": result.n_positions,
        "n_clusters": result.n_clusters,
        "not_evaluable_reason": result.not_evaluable_reason,
    }


def _direction_text(entry: Mapping, positive: str, negative: str, neither: str) -> str:
    if entry["ci_low"] is not None and entry["ci_low"] > 0:
        return positive + " (추정)"
    if entry["ci_high"] is not None and entry["ci_high"] < 0:
        return negative + " (추정)"
    return neither + " (추정)"


def band_difference_rows(
    data: JudgmentData,
    first: Callable[[str], float | None],
    second: Callable[[str], float | None],
) -> list[PositionValue]:
    """Rows of ``first - second`` per band position (report quantities)."""
    rows = []
    for pid, position in sorted(data.bands.items()):
        a, b = first(pid), second(pid)
        if a is None or b is None:
            continue
        rows.append(_position_value(position, a - b, (a + b) / 2))
    return rows


def report_band(data: JudgmentData, bootstrap: Bootstrap) -> dict:
    """R-a (24 - 18 dB), R-b (``I_px``) and R-c (textures)."""
    units = data.band_units

    def flat(pid: str, row_px: float, snr: float, recipe: str = e3x.RECIPE_R0):
        return band_arm(units, pid, row_px, recipe=recipe, snrs=(snr,))

    def rows_mean(pid: str, snr: float) -> float | None:
        return _mean_or_none(flat(pid, 47.0, snr), flat(pid, 68.0, snr))

    r_a = report_linear(
        band_difference_rows(
            data, lambda p: rows_mean(p, 24.0), lambda p: rows_mean(p, 18.0)
        ),
        Weighting.BACKGROUND,
        bootstrap,
    )
    r_a["interpretation"] = _direction_text(
        r_a, "24 dB까지 포화 없음", "24 dB에서 포화 시작", "정하지 못함"
    )
    diffs = {}
    for row_px in (90.0, 47.0):
        diffs[row_px] = band_difference_rows(
            data,
            lambda p, r=row_px: flat(p, r, 18.0, e3x.RECIPE_P42),
            lambda p, r=row_px: flat(p, r, 18.0),
        )
    by_id = {r.position_id: r for r in diffs[47.0]}
    i_px_rows = [
        replace(r, difference=r.difference - by_id[r.position_id].difference)
        for r in diffs[90.0]
        if r.position_id in by_id
    ]
    r_b = {
        "i_px": report_linear(i_px_rows, Weighting.BACKGROUND, bootstrap),
        "diff_90": report_linear(diffs[90.0], Weighting.BACKGROUND, bootstrap),
        "diff_47": report_linear(diffs[47.0], Weighting.BACKGROUND, bootstrap),
        "realized_rows": realized_rows(data.band_records),
    }
    r_c = {}
    for texture in e3x.TEXTURES:

        def textured(pid: str, t: str = texture) -> float | None:
            return _mean_or_none(
                band_arm(units, pid, 47.0, texture=t),
                band_arm(units, pid, 68.0, texture=t),
            )

        def plain(pid: str) -> float | None:
            return _mean_or_none(band_arm(units, pid, 47.0), band_arm(units, pid, 68.0))

        entry = report_linear(
            band_difference_rows(data, textured, plain), Weighting.BACKGROUND, bootstrap
        )
        entry["coh_median"] = _median_field(data.band_records, "coh", texture)
        entry["c_lat_median"] = _median_field(data.band_records, "c_lat", texture)
        r_c[texture] = entry
    r_c["none"] = {
        "coh_median": _median_field(data.band_records, "coh", e3x.TEXTURE_NONE),
        "c_lat_median": _median_field(data.band_records, "c_lat", e3x.TEXTURE_NONE),
    }
    amp = r_c[e3x.TEXTURE_AMPLITUDE]
    if amp["ci_low"] is not None:
        inside = amp["ci_low"] > -DELTA and amp["ci_high"] < DELTA
        amp["interpretation"] = (
            "AGC 뒤 실효 0.5–0.9 dB의 진폭 질감은 무관 (추정)"  # noqa: RUF001
            if inside
            else "정하지 못함 (추정)"
        )
    r_c["jitter0.2"]["interpretation"] = _direction_text(
        r_c["jitter0.2"],
        "AIHub 수준 coh의 시간 질감이 재현율을 올린다(H7 지지)",
        "시간 질감이 재현율을 내린다",
        "시간 질감 효과는 정하지 못함",
    )
    return {"R-a": r_a, "R-b": r_b, "R-c": r_c}


def _median_field(
    rows: Iterable[Mapping[str, str]], name: str, texture: str
) -> float | None:
    values = [
        float(r[name])
        for r in rows
        if _true(r["accepted"])
        and r["texture"] == texture
        and r[name] not in ("", None)
        and r["recipe"] == e3x.RECIPE_R0
        and r["length_arm"] == place.LONG_ARM
        and float(r["bin_target_px"]) in place.IN_RANGE_ROWS_PX
        and float(r["snr_db"]) in JUDGED_SNR
        and r["fold"] == str(FOLDS[0])
    ]
    finite = [v for v in values if math.isfinite(v)]
    return statistics.median(finite) if finite else None


def realized_rows(rows: Iterable[Mapping[str, str]]) -> dict:
    """Median realized truth top and compression per recipe and target row."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if not _true(r["accepted"]) or r["realized_y1_px_image"] in ("", None):
            continue
        if r["fold"] != str(FOLDS[0]) or float(r["snr_db"]) != e3x.P42_SNR_DB:
            continue
        if r["length_arm"] != place.LONG_ARM or r["texture"] != e3x.TEXTURE_NONE:
            continue
        key = f"{r['recipe']}/{float(r['bin_target_px']):g}"
        groups.setdefault(key, []).append(
            (float(r["realized_y1_px_image"]), float(r["compression"]))
        )
    return {
        key: {
            "n": len(v),
            "realized_y1_px_median": statistics.median(a for a, _ in v),
            "compression_median": statistics.median(c for _, c in v),
        }
        for key, v in sorted(groups.items())
    }


SCORE_BINS = ("no_box", "le_0.01", "0.01_0.1", "0.1_0.25", "0.25_1")


def score_bin(value: str | float | None) -> str:
    """R-d bin of a best score (``no_box`` / <= 0.01 / ... / (0.25, 1])."""
    if value in ("", None):
        return "no_box"
    number = float(value)
    if number <= 0.01:
        return "le_0.01"
    if number <= 0.1:
        return "0.01_0.1"
    if number <= 0.25:
        return "0.1_0.25"
    return "0.25_1"


def report_best_scores(
    data: JudgmentData, baseline: Sequence[Mapping[str, str]] | None
) -> dict:
    """R-d: bins of the best 0.001 score of missed units and the baseline."""
    result = {}
    for kind, rows, column, missed in (
        (
            "hyperbola",
            data.hyperbola_records,
            "best_score_001_hyp",
            lambda r: not _true(r["hit"]),
        ),
        (
            "band",
            data.band_records,
            "best_score_001_band",
            lambda r: float(r["coverage"]) == 0,
        ),
    ):
        used = [
            r
            for r in rows
            if _true(r["accepted"])
            and r.get("fold", "") != ""
            and float(r["snr_db"]) in JUDGED_SNR
            and r.get("taper", ON) == ON
            and r.get("texture", "none") == "none"
            and r.get("recipe", "r0") == "r0"
            and missed(r)
        ]
        counts = Counter(score_bin(r[column]) for r in used)
        total = sum(counts.values())
        shares = {b: (counts[b] / total if total else None) for b in SCORE_BINS}
        entry = {"n_missed_unit_folds": total, "shares": shares}
        if total:
            weak = shares["0.1_0.25"] > 0.5
            blind = shares["no_box"] + shares["le_0.01"] > 0.5
            entry["interpretation"] = (
                "약하게 봤다 (추정)"
                if weak
                else "못 봤다 (추정)"
                if blind
                else "정하지 못함 (추정)"
            )
        if baseline is not None:
            entry["baseline"] = baseline_bins(used, baseline, kind, column)
        result[kind] = entry
    if baseline is None:
        result["rd_baseline"] = None
    return result


def baseline_bins(
    used: Sequence[Mapping[str, str]],
    baseline: Sequence[Mapping[str, str]],
    kind: str,
    column: str,
) -> dict:
    """Bins of the un-injected scores at the same truths and their paired gap."""
    if kind == "hyperbola":
        index = {
            (r["position_id"], r["fold"]): r[column]
            for r in baseline
            if r["kind"] == kind
        }

        def key(r: Mapping[str, str]) -> tuple:
            return (r["position_id"], r["fold"])
    else:
        index = {
            (
                r["position_id"],
                float(r["bin_target_px"]),
                r["length_arm"],
                r["fold"],
            ): r[column]
            for r in baseline
            if r["kind"] == kind
        }

        def key(r: Mapping[str, str]) -> tuple:
            return (
                r["position_id"],
                float(r["bin_target_px"]),
                r["length_arm"],
                r["fold"],
            )

    paired = [(r[column], index[key(r)]) for r in used if key(r) in index]
    counts = Counter(score_bin(b) for _, b in paired)
    gaps = [float(a or 0.0) - float(b or 0.0) for a, b in paired]
    return {
        "n_paired": len(paired),
        "shares": {
            b: (counts[b] / len(paired) if paired else None) for b in SCORE_BINS
        },
        "mean_injected_minus_baseline": statistics.fmean(gaps) if gaps else None,
        "no_box_as_zero": True,
    }


def model_table(data: JudgmentData) -> dict:
    """R-f: ``informative`` / ``undetectable`` / ``variable`` per model (§3-5)."""
    values: dict[str, list[float]] = {}
    any_hit: dict[str, bool] = {}
    for pid, position in data.hyperbolas.items():
        value = arm_mean(data.hyperbola_units, (pid, ON), JUDGED_SNR)
        if value is not None:
            values.setdefault(position.model, []).append(value)
        any_hit.setdefault(position.model, False)
    for unit, value in data.hyperbola_units.items():
        if value > 0:
            any_hit[data.hyperbolas[unit[0]].model] = True
    table = {}
    for model in sorted(any_hit):
        v = values.get(model, [])
        if not any_hit[model]:
            label = "undetectable"
        elif len(v) >= 6 and sum(x >= 0.5 for x in v) >= 0.75 * len(v):
            label = "informative"
        else:
            label = "variable"
        table[model] = {"n_positions": len(v), "label": label}
    return table


def report_hyperbola(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> dict:
    """R-e hyperbola items: every unit, per SNR, boundary units, 6 dB, edges."""
    units, positions = data.hyperbola_units, data.hyperbolas

    def boundary_free(position: HyperbolaPosition) -> tuple[float, ...]:
        return tuple(
            s
            for s in place.edge_snr_set(position.r_edge_db)
            if abs(position.r_edge_db + s) >= 1.5
        )

    report = {
        "hh1_all_units": report_linear(
            hh1_rows(units, positions, m_star, "all"), Weighting.MODEL, bootstrap
        ),
        "hh1_12db_units": report_linear(
            hh1_rows(
                units,
                positions,
                m_star,
                snr_sets=lambda p: tuple(
                    s for s in place.edge_snr_set(p.r_edge_db) if s == 12.0
                ),
            ),
            Weighting.MODEL,
            bootstrap,
        ),
        "hh1_18db_units": report_linear(
            hh1_rows(
                units,
                positions,
                m_star,
                snr_sets=lambda p: tuple(
                    s for s in place.edge_snr_set(p.r_edge_db) if s == 18.0
                ),
            ),
            Weighting.MODEL,
            bootstrap,
        ),
        "hh1_without_boundary_units": report_linear(
            hh1_rows(units, positions, m_star, snr_sets=boundary_free),
            Weighting.MODEL,
            bootstrap,
        ),
    }
    six = [
        arm_mean(units, (pid, ON), (6.0,))
        for pid, p in positions.items()
        if p.model in m_star
    ]
    report["recall_6db_mstar_position_mean"] = (
        statistics.fmean([v for v in six if v is not None])
        if any(v is not None for v in six)
        else None
    )
    edges = Counter()
    for row in data.hyperbola_records:
        if _true(row["accepted"]) and row["n_edge_boxes"] not in ("", None):
            edges[row["taper"]] += int(row["n_edge_boxes"])
    report["edge_boxes_by_taper"] = dict(edges)
    report["negative_control_period_inj_mismatch_units"] = period_mismatch_units(
        data, m_star
    )
    return report


def period_mismatch_units(data: JudgmentData, m_star: Sequence[str]) -> int:
    """Negative-control units whose on and off ``period_inj`` differ (R-e)."""
    periods: dict[tuple, dict[str, str]] = {}
    for row in data.hyperbola_records:
        position = data.hyperbolas[row["position_id"]]
        if position.stratum != place.EdgeStratum.NEGATIVE_CONTROL.value:
            continue
        if position.model not in m_star or row["fold"] != str(FOLDS[0]):
            continue
        key = (row["position_id"], row["snr_db"], row["polarity"])
        periods.setdefault(key, {})[row["taper"]] = row["period_inj"]
    return sum(1 for v in periods.values() if len(v) == 2 and v[ON] != v[OFF])


def report_sensitivity(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> dict:
    """R-e band shifts and the R3-shared-activity subset (95 %, report only)."""
    report: dict = {}
    for name in ("HB1", "HB1b", "HB2"):
        for limit in (4.0, 8.0):
            for side, keep in (
                ("le", lambda p, cap=limit: _max_shift(p) <= cap),
                ("gt", lambda p, cap=limit: _max_shift(p) > cap),
            ):
                rows = hb_rows(name, data.band_units, data.bands, keep)
                report[f"{name}_shift_{side}_{limit:g}P"] = report_linear(
                    rows, Weighting.BACKGROUND, bootstrap
                )
    unshared_hyperbola = {
        pid: p
        for pid, p in data.hyperbolas.items()
        if p.background == inputs.DATASET_4TU and not p.shares_r3_activity
    }
    unshared_band = {
        pid: p
        for pid, p in data.bands.items()
        if p.background == inputs.DATASET_4TU and not p.shares_r3_activity
    }
    units = data.hyperbola_units
    report["r3_unshared_4tu"] = {
        "HH1": report_linear(
            hh1_rows(units, unshared_hyperbola, m_star), Weighting.MODEL, bootstrap
        ),
        "HH3": report_linear(
            hh3_rows(units, unshared_hyperbola, m_star), Weighting.MODEL, bootstrap
        ),
        **{
            name: report_linear(
                hb_rows(name, data.band_units, unshared_band),
                Weighting.BACKGROUND,
                bootstrap,
            )
            for name in ("HB1", "HB1b", "HB2")
        },
    }
    report["period_change_units_excluded"] = period_stable_report(
        data, m_star, bootstrap
    )
    report["survival_ratio"] = survival_summary(data.band_records)
    return report


def _period_stable(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Records whose realization period did not change by more than 5 %."""
    return [row for row in rows if not _true(row.get("period_changed", "False"))]


def period_stable_report(
    data: JudgmentData, m_star: Sequence[str], bootstrap: Bootstrap
) -> dict:
    """R-e: the linear hypotheses without the units whose period changed > 5 %.

    A unit drops when its realization's period changed (``[ZD]``
    ``period_changed``); a position then drops when it misses a unit.
    """
    hyperbola = fold_means(_period_stable(data.hyperbola_records), hyperbola_key, "hit")
    band = fold_means(_period_stable(data.band_records), band_key, "coverage")
    return {
        "HH1": report_linear(
            hh1_rows(hyperbola, data.hyperbolas, m_star), Weighting.MODEL, bootstrap
        ),
        "HH3": report_linear(
            hh3_rows(hyperbola, data.hyperbolas, m_star), Weighting.MODEL, bootstrap
        ),
        **{
            name: report_linear(
                hb_rows(name, band, data.bands), Weighting.BACKGROUND, bootstrap
            )
            for name in ("HB1", "HB1b", "HB2")
        },
    }


def survival_summary(rows: Iterable[Mapping[str, str]]) -> dict:
    """Median survival ratio per row x arm x SNR (fold-free, flat r0 bands)."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row["fold"] != str(FOLDS[0]) or not _true(row["accepted"]):
            continue
        if row["texture"] != e3x.TEXTURE_NONE or row["recipe"] != e3x.RECIPE_R0:
            continue
        if row["survival_ratio"] in ("", None):
            continue
        key = f"{float(row['bin_target_px']):g}/{row['length_arm']}/{row['snr_db']}"
        groups.setdefault(key, []).append(float(row["survival_ratio"]))
    return {
        key: {"n": len(values), "median": statistics.median(values)}
        for key, values in sorted(groups.items())
    }


def _max_shift(position: BandPosition) -> float:
    return max(position.max_shift_p.values(), default=0.0)


def point_estimate(rows: Sequence[PositionValue], weighting: Weighting) -> float | None:
    """Equal-weight point estimate without a bootstrap (report helpers)."""
    if not rows:
        return None
    values = np.array([r.difference for r in rows])
    weights = np.ones((1, len(rows)))
    return float(equal_weight_mean(weights, values, _groups(rows, weighting))[0])


def fold_agreement(
    data: JudgmentData, m_star: Sequence[str], estimates: Mapping[str, float | None]
) -> dict[str, int]:
    """Folds whose single-fold estimate has the sign of the judged one (R-e).

    The same rows as the judged hypotheses, with ``y_u`` of one fold only;
    a zero single-fold estimate does not agree.
    """
    agreement = dict.fromkeys(("HH1", "HH3", "HB1", "HB1b", "HB2"), 0)
    for fold in FOLDS:
        hyperbola = fold_means(data.hyperbola_records, hyperbola_key, "hit", (fold,))
        band = fold_means(data.band_records, band_key, "coverage", (fold,))
        rows = {
            "HH1": (hh1_rows(hyperbola, data.hyperbolas, m_star), Weighting.MODEL),
            "HH3": (hh3_rows(hyperbola, data.hyperbolas, m_star), Weighting.MODEL),
            **{
                name: (hb_rows(name, band, data.bands), Weighting.BACKGROUND)
                for name in ("HB1", "HB1b", "HB2")
            },
        }
        for name, (found, weighting) in rows.items():
            value, judged = point_estimate(found, weighting), estimates.get(name)
            if value is None or judged is None or value == 0:
                continue
            if math.copysign(1.0, value) == math.copysign(1.0, judged):
                agreement[name] += 1
    return agreement


# ================================================================== inputs


def assert_retry_paths(injected: Path, uninjected: Path, out_dir: Path) -> None:
    """Refuse first-try folders as inputs or output (v0.5, A-4).

    Path checks only: nothing under the refused folders is opened, listed
    or hashed. Which retry folders belong together is not decided from the
    paths; :func:`assert_run_identities` asserts one seal record and one set
    of code blobs.

    Raises:
        inputs.InputError: For a folder under ``runs/eval/m33/e3x``.
    """
    named = {"injected": injected, "uninjected": uninjected, "out": out_dir}
    first_try = [name for name, path in named.items() if e3x.is_first_try_path(path)]
    if first_try:
        raise inputs.InputError(f"first-try folder refused for {first_try}")


def _seal_identity(summary: Mapping) -> tuple[str | None, dict | None]:
    identity = summary.get("run_identity") or {}
    seal = identity.get("seal2") or {}
    return seal.get("seal_record_sha256_lf"), identity.get("code_blobs")


def assert_run_identities(
    injected_summary: Mapping, uninjected_dir: Path, record: Path
) -> None:
    """Injected and un-injected runs of the given seal record and its blobs (v5-3).

    The un-injected run is checked when its ``summary.json`` exists (it is
    optional: without it the R-d baseline is ``null``).

    Raises:
        inputs.InputError: For a run of another seal record or code blobs.
    """
    seal = json.loads(record.read_text(encoding="utf-8"))
    expected = (inputs.sha256_file_lf(record), seal.get("code_blobs"))
    if _seal_identity(injected_summary) != expected:
        raise inputs.InputError("injected run is not of the given seal record")
    summary_path = uninjected_dir / "summary.json"
    if not summary_path.is_file():
        return
    uninjected = json.loads(summary_path.read_text(encoding="utf-8"))
    if _seal_identity(uninjected) != expected:
        raise inputs.InputError("un-injected run is not of the given seal record")


UNINJECTED_RECORDS = "uninjected001_records.csv"


class BaselineSkip(StrEnum):
    """Why the un-injected run is not the R-d baseline (``rd_baseline_reason``)."""

    SUMMARY_MISSING = "summary_missing"
    STATUS_NOT_OK = "status_not_ok"
    NOT_JUDGEABLE = "not_judgeable"
    RECORDS_NOT_IN_OUTPUTS = "records_not_in_outputs_sha256"
    OUTPUT_MISSING = "output_file_missing"
    OUTPUT_CHANGED = "output_sha256_differs"


@dataclass(frozen=True)
class UninjectedBaseline:
    """The un-injected 0.001 records when they may serve as the R-d baseline.

    ``rows`` is ``None`` exactly when ``skip_reason`` is set. ``identity``
    holds the run's ``run_id``, ``summary.json`` sha256, ``status`` and
    ``judgeable`` (``None`` for what the summary does not have).
    """

    rows: list[dict[str, str]] | None
    skip_reason: BaselineSkip | None
    identity: dict


def _uninjected_identity(summary_path: Path | None, summary: Mapping) -> dict:
    return {
        "uninjected_run_id": (summary.get("run_identity") or {}).get("run_id"),
        "uninjected_summary_sha256": (
            inputs.sha256_file(summary_path) if summary_path is not None else None
        ),
        "uninjected_status": summary.get("status"),
        "uninjected_judgeable": (summary.get("judgment") or {}).get("judgeable"),
    }


def _baseline_skip_reason(
    uninjected_dir: Path, summary: Mapping
) -> BaselineSkip | None:
    # A run that did not finish ok is never opened past its summary (plan §5
    # failure table: "그 폴더는 열지 않고 남긴다"), so status and judgeable
    # are checked before any record file is hashed.
    if summary.get("status") != e3x.STATUS_OK:
        return BaselineSkip.STATUS_NOT_OK
    if (summary.get("judgment") or {}).get("judgeable") is not True:
        return BaselineSkip.NOT_JUDGEABLE
    recorded = summary.get("outputs_sha256") or {}
    if UNINJECTED_RECORDS not in recorded:
        return BaselineSkip.RECORDS_NOT_IN_OUTPUTS
    for name, expected in recorded.items():
        path = uninjected_dir / name
        if not path.is_file():
            return BaselineSkip.OUTPUT_MISSING
        if inputs.sha256_file(path) != expected:
            return BaselineSkip.OUTPUT_CHANGED
    return None


def read_uninjected_baseline(uninjected_dir: Path) -> UninjectedBaseline:
    """The R-d baseline: only an ``ok``, judgeable run with unchanged outputs.

    Any other un-injected run leaves the baseline ``null`` with the reason;
    the judged hypotheses do not use it, so nothing is refused here (plan
    §5 failure table, uninjected001 row). The seal identity of a present
    run is asserted before, by :func:`assert_run_identities`.
    """
    summary_path = uninjected_dir / "summary.json"
    if not summary_path.is_file():
        identity = _uninjected_identity(None, {})
        return UninjectedBaseline(None, BaselineSkip.SUMMARY_MISSING, identity)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = _uninjected_identity(summary_path, summary)
    reason = _baseline_skip_reason(uninjected_dir, summary)
    if reason is not None:
        return UninjectedBaseline(None, reason, identity)
    rows = inputs.read_csv(uninjected_dir / UNINJECTED_RECORDS)
    return UninjectedBaseline(rows, None, identity)


@dataclass(frozen=True)
class InjectedInputs:
    """The asserted injected run and the code blobs the judge ran with."""

    summary: dict
    hyperbola: list
    band: list
    code_blobs: dict[str, str | None]


def read_inputs(injected: Path, record: Path, plan_dir: Path) -> InjectedInputs:
    """Assert the injected summary, its files and the seal-2 record.

    Raises:
        inputs.InputError: For a summary that is not ``ok`` and judgeable,
            changed record files or a seal-2 record that differs (exit 2).
    """
    summary_path = injected / "summary.json"
    if not summary_path.is_file():
        raise inputs.InputError(f"injected summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok" or not (summary.get("judgment") or {}).get(
        "judgeable"
    ):
        raise inputs.InputError("injected run is not ok and judgeable")
    for name, expected in (summary.get("outputs_sha256") or {}).items():
        if inputs.sha256_file(injected / name) != expected:
            raise inputs.InputError(f"{name} differs from the injected summary")
    if not record.is_file():
        raise inputs.InputError(f"seal-2 record missing: {record}")
    seal = json.loads(record.read_text(encoding="utf-8"))
    if e3x.plan_file_hashes(plan_dir) != seal.get("plan_files_sha256_lf"):
        raise inputs.InputError("plan tables differ from the seal-2 record")
    blobs = m33_inputs.git_blobs(e3x.code_paths())
    recorded = seal.get("code_blobs") or {}
    changed = sorted(k for k, v in recorded.items() if blobs.get(k) != v)
    if changed or not recorded:
        raise inputs.InputError(f"code blobs differ from the seal-2 record: {changed}")
    identity = summary["run_identity"]
    if identity.get("code_blobs") != recorded:
        raise inputs.InputError("injected run code blobs differ from the seal-2 record")
    hyperbola = inputs.read_csv(injected / "hyperbola_records.csv")
    band = inputs.read_csv(injected / "band_records.csv")
    return InjectedInputs(summary, hyperbola, band, blobs)


UNIT_COLUMNS = (
    "position_id",
    "background",
    "cluster",
    "model",
    "difference",
    "floor_value",
)


def write_units(out_dir: Path, tables: Mapping[str, Sequence]) -> dict[str, str]:
    """``units_<ID>.csv`` per hypothesis (the recount basis)."""
    hashes = {}
    for name, rows in tables.items():
        path = out_dir / f"units_{name}.csv"
        if rows and isinstance(rows[0], EtaRow):
            columns = ("position_id", "background", "cluster", "model", "value")
        else:
            columns = UNIT_COLUMNS
        common.write_csv(
            path, columns, ([getattr(r, c) for c in columns] for r in rows)
        )
        hashes[path.name] = inputs.sha256_file(path)
    return hashes


def _judge_identity(
    args: argparse.Namespace,
    started_at: str,
    injected: InjectedInputs,
    baseline: UninjectedBaseline,
) -> dict:
    """The ``identity`` block; ``finished_at`` is filled when the run ends."""
    return {
        **common.new_run_identity(),
        "started_at": started_at,
        "code_blobs": injected.code_blobs,
        "injected_run_id": injected.summary["run_identity"]["run_id"],
        "injected_summary_sha256": inputs.sha256_file(args.injected / "summary.json"),
        **baseline.identity,
        "seal_record_sha256_lf": inputs.sha256_file_lf(args.seal_record),
    }


def run(args: argparse.Namespace, started_at: str | None = None) -> dict:
    """Judge the injected run; returns the ``judgment.json`` document.

    Args:
        args: The parsed CLI arguments (tests may add ``n_boot``).
        started_at: Local start time taken at :func:`main` entry; ``None``
            takes it now (direct calls).
    """
    if started_at is None:
        started_at = common.local_now()
    out_dir: Path = args.out
    assert_retry_paths(args.injected, args.uninjected, out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise inputs.InputError(f"output exists: {out_dir}")
    injected = read_inputs(args.injected, args.seal_record, args.plan)
    summary, hyperbola, band = injected.summary, injected.hyperbola, injected.band
    assert_run_identities(summary, args.uninjected, args.seal_record)
    uninjected = read_uninjected_baseline(args.uninjected)
    if uninjected.skip_reason is not None:
        logger.warning("no R-d baseline: %s", uninjected.skip_reason.value)
    identity = _judge_identity(args, started_at, injected, uninjected)
    planned = planned_band_counts(args.plan)
    baseline = uninjected.rows
    data = load_data(hyperbola, band, planned)
    n_boot = getattr(args, "n_boot", N_BOOT)  # the CLI has no option: always B
    bootstrap = Bootstrap(BOOT_SEED, n_boot)
    results, tables = judge_hypotheses(data, bootstrap)
    m_star = results["HH1"]["m_star"]
    report = report_band(data, bootstrap)
    report["R-d"] = report_best_scores(data, baseline)
    skip_reason = uninjected.skip_reason
    report["R-d"]["rd_baseline_reason"] = skip_reason.value if skip_reason else None
    report["R-e"] = {
        **report_hyperbola(data, m_star, bootstrap),
        **report_sensitivity(data, m_star, bootstrap),
        "zd_summary": summary.get("zd_summary"),
        "fold_direction_agreement": fold_agreement(
            data, m_star, {k: v.get("estimate") for k, v in results.items()}
        ),
    }
    report["R-f"] = model_table(data)
    s3_data = load_data(hyperbola, band, planned, "hit_s3", "coverage_s3")
    s3_results, _ = judge_hypotheses(s3_data, bootstrap)
    report["R-e"]["s3_label_changes"] = sum(
        s3_results[n]["decision"] != results[n]["decision"] for n in HYPOTHESES
    )
    report["R-e"]["s3_decisions"] = {n: s3_results[n]["decision"] for n in HYPOTHESES}
    report["R-e"]["s3_estimates"] = {
        n: {k: s3_results[n].get(k) for k in ("estimate", "ci_low", "ci_high")}
        for n in HYPOTHESES
    }
    plan_report = args.plan / "plan_report.json"
    if plan_report.is_file():
        plan_document = json.loads(plan_report.read_text(encoding="utf-8"))
        report["R-e"]["shift_P_by_row_model"] = plan_document["band"].get(
            "shift_P_by_row_model"
        )
    report["R-g"] = {
        name: {
            "ci_percentile_low": results[name].get("ci_percentile_low"),
            "ci_percentile_high": results[name].get("ci_percentile_high"),
            "nu": results[name].get("nu"),
            "v_by_background": results[name].get("v_by_background"),
            "g_eff_by_background": results[name].get("g_eff_by_background"),
        }
        for name in HYPOTHESES
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_hashes = write_units(out_dir, tables)
    return {
        "schema": SCHEMA,
        "status": "ok",
        "judgeable": True,
        "hypotheses": results,
        "constants": {
            "CI_LEVEL": CI_LEVEL,
            "TAIL": TAIL,
            "REPORT_CI_LEVEL": REPORT_CI_LEVEL,
            "DELTA": DELTA,
            "FLOOR": FLOOR,
            "NEG_FAIL": NEG_FAIL,
            "NEG_FLAG": NEG_FLAG,
            "NEG_MIN_POSITIONS": NEG_MIN_POSITIONS,
            "NEG_MIN_CLUSTERS": NEG_MIN_CLUSTERS,
            "SHIFT_LIMIT_P": SHIFT_LIMIT_P,
            "N_BOOT": n_boot,
            "BOOT_SEED": BOOT_SEED,
        },
        "n_boot_is_sealed": n_boot == N_BOOT,
        "report_only": report,
        "units_sha256": unit_hashes,
        "planned_band_positions": planned,
        "identity": {**identity, "finished_at": common.local_now()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; returns the exit code (module docstring)."""
    # Taken first so that identity.started_at is the real start; the identity
    # block itself is completed inside run() after the inputs are asserted.
    started = time.time()
    started_at = common.local_now()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--injected", type=Path, default=DEFAULT_INJECTED)
    parser.add_argument("--uninjected", type=Path, default=DEFAULT_UNINJECTED)
    parser.add_argument("--plan", type=Path, default=e3x.SEALED_PLAN_DIR)
    parser.add_argument("--seal-record", type=Path, default=e3x.SEAL2_RECORD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    status, n_hypotheses = "error", 0
    try:
        document = run(args, started_at)
        common.write_json(args.out / "judgment.json", document)
        status, n_hypotheses = "ok", len(document["hypotheses"])
    except inputs.InputError as error:
        logger.error("input error: %s", error)
        status = "input_error"
    except Exception:
        logger.exception("internal error")
    final = {
        "status": status,
        "out_dir": str(args.out),
        "n_hypotheses": n_hypotheses,
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
