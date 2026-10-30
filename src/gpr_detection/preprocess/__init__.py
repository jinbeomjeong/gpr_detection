"""Signal preprocessing and conversion of radargrams into detection images.

Modules: :mod:`.to_image` (the ``ImageGridMap`` coordinate contract, M1-8),
:mod:`.filters` (fitting-branch steps (1)-(4) and detection-branch ``agc``,
M2-1) and :mod:`.signals` (pick primitives shared by time zero and apex
picks). Time-zero detection (``timezero``, M2-2) is added after its approval.
"""
