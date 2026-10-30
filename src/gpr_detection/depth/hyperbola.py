"""Straight-ray hyperbola fit with a fixed pipe radius (spec_depth_v1 section 6).

All times are relative to the depth zero, ``tau = t - t_z`` (ns); positions are
along the profile, ``x = trace * dx`` (m).

* :func:`ransac_init`: robust start from ``(tau/2)^2 = a + b xi + c xi^2``.
* :func:`fit_hyperbola`: ``least_squares`` on ``(x0, t0, v)`` with ``R`` fixed.
  Forbidden for air-coupled data: the air-gap refraction flattens the wings and
  a straight-ray fit gave v'/v 1.82 with a small RMS (section 12 table 3).
* :func:`apex_vertex`: fixed-v apex time from a parabola near the apex (6-4).
* :func:`fit_curvature` / :func:`fixed_velocity_apex`: the same steps applied
  to :class:`~gpr_detection.depth.picks.HyperbolaPicks`, returning objects the
  velocity rules and ``DepthResult`` (M2-4) build on.

Velocity choice (section 7), depth (section 8) and the combined uncertainty
(section 9) are not decided here.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression, RANSACRegressor

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG, DepthConfig
from gpr_detection.depth.picks import HyperbolaPicks
from gpr_detection.io.radargram import SPEED_OF_LIGHT_M_PER_NS

V_MIN_M_PER_NS = 0.03
"""Lower velocity bound of the fit (eps_r 100)."""

V_MAX_M_PER_NS = 0.2998
"""Upper velocity bound of the fit (just below c)."""

V_FALLBACK_M_PER_NS = 0.1
"""Start velocity when RANSAC is rejected and no metadata velocity exists."""

INLIER_F_SCALES = 3.0
"""Residuals below ``3 f_scale`` are inliers (``inlier_frac``, covariance)."""

_BOUND_TOLERANCE = 1e-4
"""A parameter within this fraction of its range from a bound counts as stuck."""


def hyperbola_time(
    x_m: np.ndarray, x0_m: float, t0_ns: float, v: float, radius_m: float
) -> np.ndarray:
    """Two-way time to the cylinder top, ``tau(x)`` of section 6-1 (ns).

    ``tau = (2/v) (sqrt((v t0 / 2 + R)^2 + (x - x0)^2) - R)``; ``R = 0`` is a
    point scatterer.
    """
    centre_depth = v * t0_ns / 2.0 + radius_m
    return (2.0 / v) * (np.sqrt(centre_depth**2 + (x_m - x0_m) ** 2) - radius_m)


def _hyperbola_jacobian(
    params: np.ndarray, x_m: np.ndarray, radius_m: float
) -> np.ndarray:
    """``d tau / d (x0, t0, v)`` for every pick, shape ``[n, 3]``."""
    x0_m, t0_ns, v = params
    centre_depth = v * t0_ns / 2.0 + radius_m
    offset = x_m - x0_m
    slant = np.sqrt(centre_depth**2 + offset**2)
    d_x0 = -(2.0 / v) * offset / slant
    d_t0 = centre_depth / slant
    d_v = -(2.0 / v**2) * (slant - radius_m) + (2.0 / v) * (centre_depth / slant) * (
        t0_ns / 2.0
    )
    return np.column_stack([d_x0, d_t0, d_v])


@dataclass(frozen=True)
class RansacInit:
    """Start values from the RANSAC parabola in ``x^2 - t^2`` (section 6-2).

    Attributes:
        x0_m, t0_ns, v_m_per_ns: Start values.
        inlier_mask: RANSAC inliers (all False when RANSAC did not run).
        accepted: False when the parabola was rejected and the fallback
            ``(x_apex, tau_apex, v_fallback)`` is returned instead.
    """

    x0_m: float
    t0_ns: float
    v_m_per_ns: float
    inlier_mask: np.ndarray
    accepted: bool


def ransac_init(
    x_m: np.ndarray,
    tau_ns: np.ndarray,
    x_apex_m: float,
    tau_apex_ns: float,
    v_fallback: float | None = None,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> RansacInit:
    """RANSAC fit of ``(tau/2)^2 = a + b xi + c xi^2``, ``xi = x - x_apex``.

    ``v = 1/sqrt(c)``, ``x0 = x_apex - b/(2c)``, ``t0 = 2 sqrt(a - b^2/(4c))``.
    Rejected (``c <= 0``, ``v`` outside ``[0.03, c]`` or a negative vertex) ->
    ``(x_apex, tau_apex, v_fallback or 0.1)``.

    Args:
        x_m: Pick positions.
        tau_ns: Pick times after the zero.
        x_apex_m: Apex seed position.
        tau_apex_ns: Apex seed time after the zero.
        v_fallback: Velocity used on rejection (metadata velocity if known).
        config: ``ransac_dt_ns``, ``ransac_trials`` and ``seed``.
    """
    fallback = RansacInit(
        x0_m=x_apex_m,
        t0_ns=tau_apex_ns,
        v_m_per_ns=V_FALLBACK_M_PER_NS if v_fallback is None else v_fallback,
        inlier_mask=np.zeros(len(x_m), dtype=bool),
        accepted=False,
    )
    if len(x_m) < 3:
        return fallback

    xi = np.asarray(x_m, dtype=float) - x_apex_m
    features = np.column_stack([xi, xi**2])
    target = (np.asarray(tau_ns, dtype=float) / 2.0) ** 2
    # Residual threshold in (ns/2)^2: d(tau^2/4) = tau/2 * d tau, d tau = 0.3 ns.
    threshold = (tau_apex_ns / 2.0) * config.ransac_dt_ns
    regressor = RANSACRegressor(
        LinearRegression(),
        min_samples=3,
        residual_threshold=threshold,
        max_trials=config.ransac_trials,
        random_state=config.seed,
    )
    try:
        regressor.fit(features, target)
    except ValueError:
        # sklearn raises when no trial reaches a valid consensus set.
        return fallback

    a = float(regressor.estimator_.intercept_)
    b, c = (float(value) for value in regressor.estimator_.coef_)
    if c <= 0:
        return fallback
    v = 1.0 / np.sqrt(c)
    vertex = a - b**2 / (4.0 * c)
    if not (V_MIN_M_PER_NS <= v <= SPEED_OF_LIGHT_M_PER_NS) or vertex <= 0:
        return fallback
    return RansacInit(
        x0_m=x_apex_m - b / (2.0 * c),
        t0_ns=2.0 * float(np.sqrt(vertex)),
        v_m_per_ns=float(v),
        inlier_mask=np.asarray(regressor.inlier_mask_, dtype=bool),
        accepted=True,
    )


@dataclass(frozen=True)
class HyperbolaFit:
    """Result of the fixed-radius straight-ray fit (section 6-3).

    Attributes:
        x0_m, t0_ns, v_m_per_ns: Fitted apex position, apex time after the
            zero, velocity.
        sigma_x0_m, sigma_t0_ns, sigma_v: Standard errors from
            ``s^2 (J^T J)^+`` over inliers, ``s^2 = sum_in r^2 / (n_in - 3)``;
            NaN with three or fewer points.
        radius_m: The fixed radius used.
        rms_ns: RMS of the raw residuals.
        inlier_frac: Fraction of ``|r| < 3 f_scale``.
        n_picks: Number of picks fitted.
        converged: ``least_squares`` success.
        at_bound: A parameter ended on its bound (treated as a failure).
    """

    x0_m: float
    t0_ns: float
    v_m_per_ns: float
    sigma_x0_m: float
    sigma_t0_ns: float
    sigma_v: float
    radius_m: float
    rms_ns: float
    inlier_frac: float
    n_picks: int
    converged: bool
    at_bound: bool

    @property
    def is_usable(self) -> bool:
        """Converged and not stuck on a bound (section 10)."""
        return self.converged and not self.at_bound

    @property
    def depth_top_m(self) -> float:
        """``d_top = v t0 / 2`` (zero offset)."""
        return self.v_m_per_ns * self.t0_ns / 2.0


def _refuse_air_coupled(air_gap: str) -> None:
    """Section 7 rule 2: no straight-ray curvature fit on air-coupled data."""
    if air_gap == "air":
        raise ValueError(
            "straight-ray curvature fit is forbidden for air-coupled data; "
            "use the metadata velocity (meta_fixed) or refraction_qc"
        )


def _start_inside(start: list[float], lower: list[float], upper: list[float]):
    """Clip the start point strictly inside the bounds (least_squares needs it)."""
    span = np.subtract(upper, lower)
    return np.clip(start, np.add(lower, 1e-6 * span), np.subtract(upper, 1e-6 * span))


def _standard_errors(
    jacobian: np.ndarray, residuals: np.ndarray, inlier_limit: float
) -> np.ndarray:
    """``sqrt(diag(s^2 (J^T J)^+))`` over inliers ``|r| < inlier_limit`` (v1.3).

    Raw residuals of outliers would turn the outlier size into fit
    uncertainty, so ``s^2`` and ``J`` use inliers only; with three or fewer
    inliers all points are used. Three or fewer points in total give NaN.
    """
    n_params = jacobian.shape[1]
    inliers = np.abs(residuals) < inlier_limit
    if inliers.sum() > n_params:
        jacobian, residuals = jacobian[inliers], residuals[inliers]
    n_points = len(residuals)
    if n_points <= n_params:
        return np.full(n_params, np.nan)
    variance = float(residuals @ residuals) / (n_points - n_params)
    covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def fit_hyperbola(
    x_m: np.ndarray,
    tau_ns: np.ndarray,
    *,
    radius_m: float,
    init: RansacInit,
    air_gap: str,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> HyperbolaFit:
    """Robust ``least_squares`` fit of ``(x0, t0, v)`` with ``R`` fixed.

    Args:
        x_m: Pick positions.
        tau_ns: Pick times after the zero.
        radius_m: Fixed pipe radius (0 when unknown); never a free parameter.
        init: Start values from :func:`ransac_init`.
        air_gap: The radargram's coupling; ``"air"`` is refused.
        config: ``loss`` and ``f_scale_ns``.

    Raises:
        ValueError: For air-coupled data (spec_depth_v1 section 7 rule 2), or
            fewer than three picks.
    """
    _refuse_air_coupled(air_gap)
    x = np.asarray(x_m, dtype=float)
    tau = np.asarray(tau_ns, dtype=float)
    if len(x) < 3:
        raise ValueError(f"need at least 3 picks, got {len(x)}")

    lower = [float(x.min()), 0.0, V_MIN_M_PER_NS]
    upper = [float(x.max()), float(tau.max()), V_MAX_M_PER_NS]
    start = _start_inside([init.x0_m, init.t0_ns, init.v_m_per_ns], lower, upper)
    solution = least_squares(
        lambda params: hyperbola_time(x, *params, radius_m) - tau,
        start,
        jac=lambda params: _hyperbola_jacobian(params, x, radius_m),
        bounds=(lower, upper),
        loss=config.loss,
        f_scale=config.f_scale_ns,
        x_scale="jac",
    )
    x0_m, t0_ns, v = (float(value) for value in solution.x)
    residuals = hyperbola_time(x, x0_m, t0_ns, v, radius_m) - tau
    inlier_limit = INLIER_F_SCALES * config.f_scale_ns
    sigma = _standard_errors(
        _hyperbola_jacobian(solution.x, x, radius_m), residuals, inlier_limit
    )
    span = np.subtract(upper, lower)
    distance_to_bound = np.minimum(solution.x - lower, np.subtract(upper, solution.x))
    return HyperbolaFit(
        x0_m=x0_m,
        t0_ns=t0_ns,
        v_m_per_ns=v,
        sigma_x0_m=float(sigma[0]),
        sigma_t0_ns=float(sigma[1]),
        sigma_v=float(sigma[2]),
        radius_m=radius_m,
        rms_ns=float(np.sqrt(np.mean(residuals**2))),
        inlier_frac=float(np.mean(np.abs(residuals) < inlier_limit)),
        n_picks=len(x),
        converged=bool(solution.success),
        at_bound=bool(np.any(distance_to_bound <= _BOUND_TOLERANCE * span)),
    )


@dataclass(frozen=True)
class ApexVertex:
    """Fixed-velocity apex from a parabola near the seed (section 6-4).

    Attributes:
        x0_m: Vertex position (the seed position on fallback).
        t0_ns: Vertex time after the zero (the seed time on fallback).
        n_points: Points used in the parabola (1 on fallback).
        rms_ns: RMS of the parabola residuals (0 on fallback).
        used_fallback: True when the seed was used directly.
    """

    x0_m: float
    t0_ns: float
    n_points: int
    rms_ns: float
    used_fallback: bool


def apex_vertex(
    x_m: np.ndarray,
    tau_ns: np.ndarray,
    x_apex_m: float,
    tau_apex_ns: float,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> ApexVertex:
    """Fit ``tau = p0 + p1 xi + p2 xi^2`` within ``apex_halfwidth_m`` of the seed.

    ``t0 = p0 - p1^2 / (4 p2)``, ``x0 = x_apex - p1 / (2 p2)``. Falls back to
    the seed when fewer than ``apex_min_points`` points exist, ``p2 <= 0``, or
    the vertex lands outside the fitted window.
    """
    fallback = ApexVertex(x_apex_m, tau_apex_ns, 1, 0.0, True)
    xi = np.asarray(x_m, dtype=float) - x_apex_m
    near = np.abs(xi) <= config.apex_halfwidth_m + 1e-9
    if near.sum() < config.apex_min_points:
        return fallback
    tau_near = np.asarray(tau_ns, dtype=float)[near]
    p2, p1, p0 = np.polyfit(xi[near], tau_near, 2)
    if p2 <= 0:
        return fallback
    xi_vertex = -p1 / (2.0 * p2)
    if abs(xi_vertex) > config.apex_halfwidth_m:
        return fallback
    residuals = np.polyval([p2, p1, p0], xi[near]) - tau_near
    return ApexVertex(
        x0_m=x_apex_m + float(xi_vertex),
        t0_ns=float(p0 - p1**2 / (4.0 * p2)),
        n_points=int(near.sum()),
        rms_ns=float(np.sqrt(np.mean(residuals**2))),
        used_fallback=False,
    )


class FitStatus(StrEnum):
    """Outcome of :func:`fit_curvature` (names shared with ``DepthResult.status``)."""

    OK = "ok"
    INSUFFICIENT_PICKS = "insufficient_picks"
    FIT_FAILED = "fit_failed"


@dataclass(frozen=True)
class CurvatureFit:
    """Curvature (free-v) fit of one box's picks, before any velocity rule.

    Attributes:
        status: ``ok``, ``insufficient_picks`` (section 5-4) or ``fit_failed``
            (no apex seed, or ``least_squares`` not converged or stuck on a
            bound, section 10).
        init: RANSAC start values; ``None`` when the fit was not attempted.
        fit: The ``least_squares`` result; ``None`` when not attempted.
        flags: ``ransac_rejected``; without an apex, the picks' flags
            (``box_in_zero_zone``, ``no_apex_seed``, ...).
    """

    status: FitStatus
    init: RansacInit | None
    fit: HyperbolaFit | None
    flags: tuple[str, ...]

    @property
    def is_ok(self) -> bool:
        """True when ``fit`` can be used for a curvature velocity."""
        return self.status == FitStatus.OK


def fit_curvature(
    picks: HyperbolaPicks,
    *,
    radius_m: float,
    air_gap: str,
    v_fallback: float | None = None,
    config: DepthConfig = DEFAULT_DEPTH_CONFIG,
) -> CurvatureFit:
    """RANSAC start + fixed-radius ``least_squares`` on scaled picks (6-2, 6-3).

    Args:
        picks: Picks with a physical scale and a time zero.
        radius_m: Fixed pipe radius (known outer diameter / 2, else 0).
        air_gap: The radargram's coupling; ``"air"`` raises ``ValueError``.
        v_fallback: Start velocity if RANSAC is rejected (metadata v if known).
        config: Pick-count rule, RANSAC and fit settings.

    Raises:
        ValueError: For air-coupled data, or picks without scale or zero.
    """
    # Refuse before the pick count so the answer does not depend on the data.
    _refuse_air_coupled(air_gap)
    if picks.apex is None:
        # No apex at all: no velocity can give a depth (section 10, v1.3).
        return CurvatureFit(FitStatus.FIT_FAILED, None, None, picks.flags)
    if not picks.is_enough_for_curvature(config):
        return CurvatureFit(FitStatus.INSUFFICIENT_PICKS, None, None, ())

    x_m, tau_ns = picks.x_m, picks.tau_ns
    init = ransac_init(
        x_m, tau_ns, picks.apex_x_m, picks.apex_tau_ns, v_fallback, config
    )
    fit = fit_hyperbola(
        x_m, tau_ns, radius_m=radius_m, init=init, air_gap=air_gap, config=config
    )
    flags = () if init.accepted else ("ransac_rejected",)
    status = FitStatus.OK if fit.is_usable else FitStatus.FIT_FAILED
    return CurvatureFit(status, init, fit, flags)


def fixed_velocity_apex(
    picks: HyperbolaPicks, config: DepthConfig = DEFAULT_DEPTH_CONFIG
) -> ApexVertex:
    """Section 6-4 apex vertex of scaled picks (used whenever v is fixed).

    Raises:
        ValueError: Without an apex seed, physical scale or time zero.
    """
    return apex_vertex(
        picks.x_m, picks.tau_ns, picks.apex_x_m, picks.apex_tau_ns, config
    )
