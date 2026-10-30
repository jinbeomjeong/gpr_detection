"""Settings of hyperbola depth estimation in one place (spec_depth_v1 section 13).

The defaults are the spec values; the fields beyond the spec list
(``box_expand_frac`` ...) name constants that the spec gives in the text.
"""

from dataclasses import dataclass

from gpr_detection.preprocess.filters import BACKGROUND_WINDOW_TRACES
from gpr_detection.preprocess.signals import PickMode
from gpr_detection.preprocess.timezero import GroundZeroShift, ZeroConfig


@dataclass(frozen=True)
class DepthConfig:
    """Hyperbola depth settings (spec_depth_v1 sections 4-9).

    Attributes:
        a_max_m: Max half-aperture of the pick tracking (section 5-3).
        seed_rel: First-peak threshold for the apex seed, vs the box maximum.
        track_stop_rel: Tracking stops below this fraction of the apex value.
        v_min: Slowest velocity used to size the tracking window (m/ns).
        ransac_dt_ns: RANSAC residual threshold expressed as a time (ns).
        ransac_trials: RANSAC ``max_trials``.
        f_scale_ns: ``least_squares`` soft-L1 scale (ns).
        apex_halfwidth_m: Half-width of the fixed-v apex parabola (section 6-4).
        bg_window: Background median window in traces.
        h_max_m: Max antenna height searched for the surface reflection.
        ground_zero_shift: Optional direct-wave delay correction (off).
        sigma_eps_rel: Assumed relative uncertainty of a metadata eps_r.
        seed: RANSAC random state.
        pick_mode: Phase reference shared by the zero and the apex.
        box_expand_frac: Box widening per side along traces (section 5-1).
        track_back_ns: Tracking window starts this much before the previous pick.
        track_reverse_ns: Moving outwards, an earlier pick by more than this stops.
        min_picks: Minimum picks for a curvature fit (section 5-4).
        min_side_picks: Minimum picks on each side of the apex.
        apex_edge_traces: Apex this close to a box side gets ``apex_at_box_edge``.
        apex_min_points: Minimum points of the fixed-v apex parabola.
        loss: ``least_squares`` loss, ``"soft_l1"`` or ``"huber"``.
        curvature_sigma_floor: Relative v uncertainty added to curvature fits.
        refraction_eps_min: Lower end of the refraction QC eps_r grid.
        refraction_eps_max: Upper end of the refraction QC eps_r grid.
        refraction_eps_step: Grid step of the refraction QC.
        meta_eps_suspect_rel: Refraction QC vs meta eps_r beyond this flags it.
    """

    a_max_m: float = 1.0
    seed_rel: float = 0.3
    track_stop_rel: float = 0.2
    v_min: float = 0.05
    ransac_dt_ns: float = 0.3
    ransac_trials: int = 200
    f_scale_ns: float = 0.1
    apex_halfwidth_m: float = 0.10
    bg_window: int = BACKGROUND_WINDOW_TRACES
    h_max_m: float = 0.5
    ground_zero_shift: GroundZeroShift = GroundZeroShift.NONE
    sigma_eps_rel: float = 0.15
    seed: int = 0
    pick_mode: PickMode = "envelope"
    box_expand_frac: float = 0.1
    track_back_ns: float = 0.3
    track_reverse_ns: float = 0.1
    min_picks: int = 9
    min_side_picks: int = 3
    apex_edge_traces: int = 3
    apex_min_points: int = 5
    loss: str = "soft_l1"
    curvature_sigma_floor: float = 0.03
    refraction_eps_min: float = 3.0
    refraction_eps_max: float = 30.0
    refraction_eps_step: float = 0.25
    meta_eps_suspect_rel: float = 0.3

    def zero_config(self) -> ZeroConfig:
        """The time-zero settings implied by these depth settings."""
        return ZeroConfig(
            h_max_m=self.h_max_m,
            window_traces=self.bg_window,
            ground_zero_shift=self.ground_zero_shift,
        )


DEFAULT_DEPTH_CONFIG = DepthConfig()
