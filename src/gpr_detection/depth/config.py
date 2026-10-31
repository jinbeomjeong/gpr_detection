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
        candidate_rel: First-peak threshold for the apex candidate, vs the box maximum.
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
        meta_eps_suspect_rel: Metadata eps_r below ``(1 - this) eps_lo`` or above
            ``(1 + this) eps_hi`` of the refraction QC interval flags it.
        refraction_h_margin_m: The QC interval also runs at ``h_est`` minus this
            (direct-wave interference biases ``h_est`` high; assumed, v1.4).
        hint_traces: Apex-hint window half-width in traces (section 5-7, v1.8).
        hint_periods: Apex-hint window half-height in periods ``1 / f_c``
            (no scale: ``max(2, hint_periods * P_px)`` px).
        top_guard_samples: An integer peak index ``k_p - k_top`` at or below this
            is rejected as a box-top boundary peak (section 5-7 item 2).
        top_guard_px: The same guard without a physical scale (pixels).
        hint_noise_k: Hint-window peaks must reach ``k sigma_n`` (``1.4826 MAD``
            of the background-removed box below the search top; v1.8 supplement).
        hint_box_rel_floor: Hint-window peaks must also reach this fraction of
            the widened-box maximum (-26 dB); it matters only on noise-free data
            where ``sigma_n`` is 0.
    """

    a_max_m: float = 1.0
    candidate_rel: float = 0.3
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
    refraction_h_margin_m: float = 0.06
    hint_traces: int = 3
    hint_periods: float = 0.5
    top_guard_samples: int = 2
    top_guard_px: int = 1
    hint_noise_k: float = 5.0
    hint_box_rel_floor: float = 0.05

    def zero_config(self) -> ZeroConfig:
        """The time-zero settings implied by these depth settings."""
        return ZeroConfig(
            h_max_m=self.h_max_m,
            window_traces=self.bg_window,
            ground_zero_shift=self.ground_zero_shift,
        )


DEFAULT_DEPTH_CONFIG = DepthConfig()


@dataclass(frozen=True)
class BandConfig:
    """``pipe_band`` settings (spec_band_depth_v1 section 8-2).

    Lives here, not in ``band.py``, so that ``velocity.py`` (band velocity
    rules) and ``band.py`` can both use it without importing each other.

    Attributes:
        candidate_rel: First-peak threshold vs the box envelope maximum.
        pad_up_periods: Search window extension above the box, in ``1 / f_c``.
        pad_down_periods: Extension below the box, in ``1 / f_c``.
        bg_window: Masked-median window in traces (centred on the band).
        bg_margin_m: Trace margin around every band box (at least 5 traces).
        bg_min_outside: Fewer outside traces skip the background removal.
        merge_gap_px: Fragment gap for no-scale inputs (traces = px); v1.4,
            band spec 3-4 (was 8).
        merge_gap_m: Fragment gap for scaled inputs.
        merge_overlap_min: Required time overlap of the smaller box height.
        f_scale_ns: ``least_squares`` soft-L1 scale (1 px without scale).
        signif_sigma: Significance of the slope / curvature (in sigmas).
        slope_min_ns: Minimum time change over the pick range for ``sloped``.
        res_max_periods: ``sigma_res`` above this many periods fails.
        coverage_min: Below this coverage the band has too few picks.
        n_picks_min: Below this pick count the band has too few picks.
        oblique_flag_deg: ``theta_est`` above this flags ``band_oblique_large``.
        footprint_corr_m: Pick correlation length for ``n_eff``.
        dataset_default_eps: Assumed eps_r when no other source exists; off.
        pick_mode: Phase reference shared with the zero.
    """

    candidate_rel: float = 0.3
    pad_up_periods: float = 1.0
    pad_down_periods: float = 0.5
    bg_window: int = BACKGROUND_WINDOW_TRACES
    bg_margin_m: float = 0.1
    bg_min_outside: int = 30
    merge_gap_px: int = 32
    merge_gap_m: float = 0.2
    merge_overlap_min: float = 0.5
    f_scale_ns: float = 0.1
    signif_sigma: float = 3.0
    slope_min_ns: float = 0.2
    res_max_periods: float = 0.25
    coverage_min: float = 0.3
    n_picks_min: int = 5
    oblique_flag_deg: float = 30.0
    footprint_corr_m: float = 0.2
    dataset_default_eps: float | None = None
    pick_mode: PickMode = "envelope"


DEFAULT_BAND_CONFIG = BandConfig()
