"""Peak-picking primitives shared by time-zero detection and depth picks.

spec_depth_v1 section 0-4 requires the time zero and the apex to be picked with
the **same phase reference**. Both :mod:`.timezero` and ``depth/picks.py``
therefore go through :func:`pick_signal` and :func:`parabolic_peak` here.
"""

from typing import Literal, get_args

import numpy as np
from scipy.signal import hilbert

PickMode = Literal["envelope", "signed_max", "signed_min"]
"""Phase reference of a pick: Hilbert envelope peak (default) or signed extremum."""

PICK_MODES: tuple[str, ...] = get_args(PickMode)


def check_pick_mode(pick_mode: str) -> None:
    """Raise ``ValueError`` for anything other than a :data:`PickMode` value."""
    if pick_mode not in PICK_MODES:
        raise ValueError(f"pick_mode must be one of {PICK_MODES}, got {pick_mode!r}")


def require_same_pick_mode(zero_mode: str, apex_mode: str) -> None:
    """Refuse mixed phase references (spec_depth_v1 section 5-5, test U6).

    Raises:
        ValueError: If the time zero and the apex use different pick modes.
    """
    if zero_mode != apex_mode:
        raise ValueError(
            f"time zero was picked with {zero_mode!r} but the apex uses "
            f"{apex_mode!r}; both must share one phase reference"
        )


def pick_signal(section: np.ndarray, pick_mode: PickMode) -> np.ndarray:
    """Return the non-negative-peaked signal that picks are taken on.

    Args:
        section: Amplitudes, time along axis 0 (a trace or ``[samples, traces]``).
        pick_mode: ``"envelope"`` gives ``|hilbert(section)|`` along time,
            ``"signed_max"`` the amplitudes themselves and ``"signed_min"`` their
            negation, so that every mode picks maxima.

    Returns:
        float64 array of the same shape.
    """
    check_pick_mode(pick_mode)
    values = np.asarray(section, dtype=np.float64)
    if pick_mode == "envelope":
        return np.abs(hilbert(values, axis=0))
    if pick_mode == "signed_max":
        return values.copy()
    return -values


def parabolic_peak(values: np.ndarray, index: int) -> tuple[float, float]:
    """Refine a discrete maximum with a three-point parabola.

    Args:
        values: 1-D signal.
        index: Integer index of a local maximum.

    Returns:
        ``(fractional_index, interpolated_value)``. At the array ends, or on a
        flat top, the integer index and its value are returned unchanged.
    """
    if not 0 < index < len(values) - 1:
        return float(index), float(values[index])
    left, centre, right = values[index - 1], values[index], values[index + 1]
    curvature = left - 2.0 * centre + right
    if curvature == 0:
        return float(index), float(centre)
    offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return index + offset, float(centre - 0.25 * (left - right) * offset)
