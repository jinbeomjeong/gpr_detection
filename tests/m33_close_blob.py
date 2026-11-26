"""Stand-in for the sealed C0a ``pipeline.py`` blob in M3-3 runner tests.

The M3-3 close (user decision 2026-11-16) changed only the ``detect_line``
default of ``src/gpr_detection/pipeline.py`` to the C0b set S3; ``apply_pp``
and ``PP_SETS`` are unchanged (``tests/test_pp_default.py``). The sealed
runners keep refusing that tree (``m33_inputs.assert_sealed_pp``,
``pipeline_blob_is_c0a``; pinned by ``test_m32_recipe_eval``), so their
end-to-end fixtures - and only those - accept the working-tree blob as the
sealed one (user decision 2026-11-26, option A). The R3 change report keeps
its own copy (``m33_r3_change_report.C1_CODE_BLOBS``); its fixture swaps the
same blob in there.

The caller must have put ``analysis/detection`` on ``sys.path`` first.
"""

import pytest

CLOSE_PIPELINE_BLOB = "cbc7b84784f26f3dac1c10f62bd757fd74e7df4d"
"""``pipeline.py`` blob of the M3-3 close (only the ``detect_line`` default
changed); the stand-in is refused for any other blob."""


def working_tree_pipeline_blob() -> str:
    """``git hash-object`` of ``pipeline.py``, as the runners measure it."""
    import m33_inputs

    path = m33_inputs.REPO / m33_inputs.PIPELINE_PATH
    blob = m33_inputs.git_blobs([path])[m33_inputs.PIPELINE_PATH]
    if blob is None:
        raise RuntimeError("git hash-object of pipeline.py failed")
    return blob


def accept_close_pipeline_blob(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the working-tree ``pipeline.py`` blob the sealed one for one test.

    Raises:
        AssertionError: If ``pipeline.py`` is not the M3-3 close blob.
    """
    import m33_inputs

    blob = working_tree_pipeline_blob()
    if blob != CLOSE_PIPELINE_BLOB:
        raise AssertionError(
            f"pipeline.py blob {blob} != close {CLOSE_PIPELINE_BLOB}: "
            "pipeline.py가 M3-3 마감 뒤 바뀜 — 의도한 변경이면 이 상수를 "
            "갱신하고 결정 기록을 남긴다"
        )
    monkeypatch.setattr(m33_inputs, "C0A_PIPELINE_BLOB", blob)
    return blob
