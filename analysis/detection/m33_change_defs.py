"""M3-3 R3 report definitions: when has a re-measured zero or period changed?

Spec: ``docs/processing/m33_r3_zero_change_definition.md`` §3-§5-1 (user
decision 2026-11-20: branch B, symmetric keys, strict tolerances).

The R3 runner compared the zero re-measured on an injected record with the
line zero by exact float equality. The zero is a fractional sample, so the
first-order leak of a band through the DC removal, the filters and the
Hilbert envelope (linear in the band amplitude) counted as a change. The
definitions here count a change only when it exceeds the resolution of the
measurement itself:

* zero changed: the zero event label differs, one side failed, or
  ``|dk| > ZERO_CHANGE_TOLERANCE_P * P``;
* period changed: the injected period could not be measured, or
  ``|P_inj - P| > PERIOD_CHANGE_TOLERANCE_REL * P``.

A changed rule name or event count is reported but is not a zero change: in
R3 every such realization kept its zero event (spec §2-1, §3-2).

Both tolerances are compared in product form (``|d| > tol * P``) rather than
as a ratio (``|d / P| > tol``). The two agree except within one ulp of the
boundary; the product keeps a difference of exactly ``tol * P`` (for example
``P_inj / P = 21 / 20``) on the "not changed" side, where the ratio form
turns ``1.05 - 1`` into 0.050000000000000044. The ``exact`` fields keep the
old definition (``!=``) for the published counts.
"""

from dataclasses import dataclass

from gpr_detection.preprocess.timezero import ZeroResult

ZERO_CHANGE_TOLERANCE_P = 0.1
"""Zero tolerance in units of the line period ``P`` (spec §3-2).

0.1 P is a fifth of the narrowest downstream margin (PP-d, 0.5 P) and the
size of the zero standard uncertainty (0.15-0.25 ns) on the R3 lines.
"""
PERIOD_CHANGE_TOLERANCE_REL = 0.05
"""Relative period tolerance (spec §3-3).

The same value as ``inject.PERIOD_CHECK_TOLERANCE``, the project's "same
period" rule; the tests and the report tool assert the two are equal.
"""


@dataclass(frozen=True)
class ZeroChange:
    """How the zero of an injected record differs from the line zero.

    Attributes:
        dk_samples: ``k_inj - k_bg`` in samples; ``None`` if either failed.
        dk_over_p: ``dk_samples / P``; ``None`` if either failed.
        label_changed: Both zeros found and their event labels differ.
        failed_changed: Exactly one side failed to find a zero.
        rule_changed: Both zeros found and their rule names differ (report only).
        events_changed: The number of early events differs (report only).
        exceeds_tolerance: ``|dk_samples| > ZERO_CHANGE_TOLERANCE_P * P``.
        changed: ``label_changed or failed_changed or exceeds_tolerance``.
        exact: The old definition, ``k_inj != k_bg``.
    """

    dk_samples: float | None
    dk_over_p: float | None
    label_changed: bool
    failed_changed: bool
    rule_changed: bool
    events_changed: bool
    exceeds_tolerance: bool
    changed: bool
    exact: bool


@dataclass(frozen=True)
class PeriodChange:
    """How the period of an injected record differs from the line period.

    Attributes:
        dp_samples: ``P_inj - P`` in samples; ``None`` if either is missing.
        dp_rel: ``P_inj / P - 1``; ``None`` if either is missing.
        exceeds_tolerance: ``|P_inj - P| > PERIOD_CHANGE_TOLERANCE_REL * P``.
        failed: A period is missing (a failed measurement is a change).
        changed: ``exceeds_tolerance or failed``.
        exact: The old definition, ``P_inj != P``.
    """

    dp_samples: float | None
    dp_rel: float | None
    exceeds_tolerance: bool
    failed: bool
    changed: bool
    exact: bool


def _zero_failed(zero: ZeroResult) -> bool:
    return zero.failed or zero.k_zero is None


def zero_change(
    background: ZeroResult, injected: ZeroResult, period_samples: float
) -> ZeroChange:
    """Compare the zero re-measured on an injected record with the line zero.

    Args:
        background: Zero of the background line (``LineContext.zero``).
        injected: Zero of the injected record, measured on the same path.
        period_samples: Line period ``P`` in L0 samples (``P_line``).

    Raises:
        ValueError: If ``period_samples`` is missing or not positive.
    """
    if period_samples is None or not period_samples > 0:
        raise ValueError(f"period_samples must be positive, got {period_samples!r}")
    background_failed = _zero_failed(background)
    injected_failed = _zero_failed(injected)
    both_found = not background_failed and not injected_failed
    dk_samples = (
        float(injected.k_zero - background.k_zero)  # type: ignore[operator]
        if both_found
        else None
    )
    exceeds_tolerance = dk_samples is not None and bool(
        abs(dk_samples) > ZERO_CHANGE_TOLERANCE_P * period_samples
    )
    label_changed = both_found and (
        injected.zero_event_label != background.zero_event_label
    )
    failed_changed = background_failed != injected_failed
    return ZeroChange(
        dk_samples=dk_samples,
        dk_over_p=None if dk_samples is None else dk_samples / period_samples,
        label_changed=label_changed,
        failed_changed=failed_changed,
        rule_changed=both_found and injected.zero_rule != background.zero_rule,
        events_changed=len(injected.events_ns) != len(background.events_ns),
        exceeds_tolerance=exceeds_tolerance,
        changed=label_changed or failed_changed or exceeds_tolerance,
        exact=bool(injected.k_zero != background.k_zero),
    )


def period_change(
    background_period: float | None, injected_period: float | None
) -> PeriodChange:
    """Compare the period of an injected record's image with the line period.

    Args:
        background_period: ``P_line`` of the background line, in samples.
        injected_period: ``period_samples`` of the injected record's image.

    Raises:
        ValueError: If ``background_period`` is given but not positive.
    """
    if background_period is not None and not background_period > 0:
        raise ValueError(f"background_period must be positive: {background_period!r}")
    failed = background_period is None or injected_period is None
    if failed:
        dp_samples = dp_rel = None
        exceeds_tolerance = False
    else:
        dp_samples = float(injected_period - background_period)
        dp_rel = float(injected_period / background_period - 1.0)
        exceeds_tolerance = bool(
            abs(dp_samples) > PERIOD_CHANGE_TOLERANCE_REL * background_period
        )
    return PeriodChange(
        dp_samples=dp_samples,
        dp_rel=dp_rel,
        exceeds_tolerance=exceeds_tolerance,
        failed=failed,
        changed=exceeds_tolerance or failed,
        exact=bool(injected_period != background_period),
    )
