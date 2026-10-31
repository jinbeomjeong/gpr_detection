"""Tests of the report writing in ``scripts/check_band_picks_aihub.py``.

Only the version table and the generated-block merge are tested; the band
picks themselves are covered by ``tests/depth``.
"""

import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs" / "processing" / "spec_band_depth_v1.md"
# Group 1: version label, 2-3: major/minor, 4: rest of the bold entry heading.
SPEC_VERSION_ENTRY = re.compile(r"^- \*\*(v(\d+)\.(\d+)) (\(.*?)\*\*", re.MULTILINE)
# Marker the spec owner puts in the heading of an approved but unimplemented
# revision, e.g. "코드 변경분은 구현 대기, 현재 코드는 v1.4".
PENDING_IMPLEMENTATION_MARK = "구현 대기"
STATED_CODE_VERSION = re.compile(r"현재 코드는 (v\d+\.\d+)")


@pytest.fixture
def script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("check_band_picks_aihub")


def _row(*, flags: str, confidence: object, ratio: object) -> dict[str, object]:
    return {"flags": flags, "confidence": confidence, "sigma_res_over_period": ratio}


def test_current_column_counts_flags_and_results(script: ModuleType) -> None:
    rows = [
        _row(flags="inconsistent_picks fragment_of_group", confidence=0.0, ratio=0.5),
        _row(flags="band_concave", confidence=0.2, ratio=""),
        _row(flags="", confidence="", ratio=1.0),
    ]
    images = [
        script.ImageResult([], [], 4.0, 4.0),
        script.ImageResult([], [], None, 3.0),
    ]

    column = script.current_run_column(rows, images)

    assert column["inconsistent_picks"] == "1/3"
    assert column["band_concave"] == "1/3"
    assert column["skipped"] == "0/3 결과, 1장"
    assert column["results_and_fragments"] == "3 / 1"
    assert column["sigma_res_over_period"] == "0.50 / 0.62 / 0.75 / 0.88 / 1.00"
    assert set(column) == set(script.COMPARISON_ROW_LABELS)


def test_table_keeps_frozen_v13_column_next_to_current(script: ModuleType) -> None:
    current = dict.fromkeys(script.COMPARISON_ROW_LABELS, "X")

    lines = script.comparison_table(current, n_boxes=53)

    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    assert header == [
        "항목",
        "v1.0 (고정 1/2/2/3/1 px)",
        "v1.1 (P_px 기준, 하한 4 px)",
        "v1.3 (P_px 기준, 하한 2.2 px)",
        f"{script.CURRENT_SPEC_VERSION} ({script.CURRENT_RUN_SUMMARY}, 이번)",
    ]
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2] == "| flag `inconsistent_picks` | 30/51 | 39/51 | 42/51 | X |"
    fragments_row = "| 박스 53개 → 결과 / `fragment_of_group` | - | - | 51 / 6 | X |"
    assert lines[-1] == fragments_row


def test_only_the_current_column_is_marked_as_this_run(script: ModuleType) -> None:
    for label in script.PREVIOUS_RUNS:
        assert "이번" not in label
        assert not label.startswith(f"{script.CURRENT_SPEC_VERSION} ")


def test_current_version_is_latest_implemented_band_spec_entry(
    script: ModuleType,
) -> None:
    """``CURRENT_SPEC_VERSION`` must be the newest *implemented* spec revision.

    A change-history entry whose bold heading contains "구현 대기" is approved
    but not yet in the code, so it is skipped. If such a heading also states
    "현재 코드는 vX.Y", that version must equal ``CURRENT_SPEC_VERSION``.

    When the pending revision (v1.5: R2, R3) is implemented: the spec owner
    removes the "구현 대기" mark, ``CURRENT_SPEC_VERSION`` is raised to that
    revision, and the old current column moves into ``PREVIOUS_RUNS``.
    """
    entries = SPEC_VERSION_ENTRY.findall(SPEC_PATH.read_text(encoding="utf-8"))
    assert entries, "band spec change history not found"

    implemented = []
    for label, major, minor, heading in entries:
        if PENDING_IMPLEMENTATION_MARK not in heading:
            implemented.append((int(major), int(minor), label))
            continue
        stated = STATED_CODE_VERSION.search(heading)
        if stated:
            assert stated.group(1) == script.CURRENT_SPEC_VERSION, label

    assert implemented, "no implemented band spec revision found"
    latest_implemented = max(implemented)[2]
    assert latest_implemented == script.CURRENT_SPEC_VERSION


def test_merge_writes_markers_for_a_new_report(script: ModuleType) -> None:
    text = script.merge_generated_block(None, "# body")

    assert text == f"{script.GENERATED_BEGIN}\n# body\n{script.GENERATED_END}\n"


def test_merge_keeps_hand_written_text_outside_markers(script: ModuleType) -> None:
    existing = (
        f"intro\n{script.GENERATED_BEGIN}\nold body\n{script.GENERATED_END}\n"
        "\n## hand notes\n- keep me\n"
    )

    text = script.merge_generated_block(existing, "new body")

    assert text == (
        f"intro\n{script.GENERATED_BEGIN}\nnew body\n{script.GENERATED_END}\n"
        "\n## hand notes\n- keep me\n"
    )


@pytest.mark.parametrize(
    "existing",
    [
        "no markers at all\n",
        "{begin}\nonly begin\n",
        "{end}\n{begin}\nreversed\n",
        "{begin}\na\n{end}\n{begin}\nb\n{end}\n",
    ],
)
def test_merge_refuses_reports_without_one_marker_pair(
    script: ModuleType, existing: str
) -> None:
    existing = existing.format(begin=script.GENERATED_BEGIN, end=script.GENERATED_END)

    with pytest.raises(ValueError, match="marker"):
        script.merge_generated_block(existing, "new body")
