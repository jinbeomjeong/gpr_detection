"""Evaluation-only tools (architecture README section 2-2, rules 8 and 9).

Modules: :mod:`.inject` (M3-2 S4: gprMax pipe responses injected into real
background lines, with L0 truth boxes and the seeded injection table).

Rule 8: this package imports only the standard library, numpy, scipy,
``gpr_detection.io`` and ``gpr_detection.preprocess``. Rule 9: no module of
``gpr_detection`` outside this package imports it; it never enters the
training or inference path. Nothing is imported here on purpose, so
``import gpr_detection.eval`` stays cheap.
"""
