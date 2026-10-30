"""Physics-based pipe depth from GPR B-scans (ADR-002, spec_depth_v1).

Receives only L0 grid boxes (``GridBox``) and never imports ``detect``
(architecture README section 2-2). Modules: :mod:`.config` (settings),
:mod:`.picks` (apex candidate and tracking), :mod:`.hyperbola` (RANSAC start,
fixed-radius fit, fixed-v apex), :mod:`.refraction` (air-gap ray model and
refraction QC), :mod:`.velocity` (section 7 velocity rules), :mod:`.result`
(``DepthResult``, ``BandGeometry``), :mod:`.estimate` (``estimate_depth``: one
box, sections 8-10) and :mod:`.band` (``estimate_band_depths``: ``pipe_band``
boxes, spec_band_depth_v1).
"""
