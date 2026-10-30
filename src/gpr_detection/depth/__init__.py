"""Physics-based pipe depth from GPR B-scans (ADR-002, spec_depth_v1).

Receives only L0 grid boxes (``GridBox``) and never imports ``detect``
(architecture README section 2-2). Modules so far (M2-3): :mod:`.config`
(settings), :mod:`.picks` (apex seed and tracking) and :mod:`.hyperbola`
(RANSAC start, fixed-radius fit, fixed-v apex). The refraction model, velocity
rules, ``DepthResult`` and uncertainty follow in M2-4.
"""
