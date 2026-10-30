"""Signal preprocessing and conversion of radargrams into detection images.

Modules: :mod:`.to_image` (the ``ImageGridMap`` coordinate contract, M1-8),
:mod:`.filters` (fitting-branch steps (1)-(4) and detection-branch ``agc``,
M2-1), :mod:`.signals` (pick primitives shared by time zero and apex picks)
and :mod:`.timezero` (time-zero events, rules and window QC, M2-2).
"""
