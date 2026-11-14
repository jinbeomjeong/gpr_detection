"""M3-2 S5: fixed inputs of the stage-C recipe runner and judge.

Sources: plan ``docs/processing/m32_image_quality_plan.md`` v0.2.3 sections
4-1 (recipes), 4-3 (thresholds), 4-7 (stage-A lists, K bundle and their
sha256) and ``docs/experiments/m32_cv_review.md`` section 4-2.

Every file the stage-C result depends on is named here with the sha256 the
plan fixed; the runner asserts them before it loads a model (exit 2 on a
mismatch).
"""

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from gpr_detection.io.radargram import GridBox
from gpr_detection.preprocess.to_image import DetectionImageConfig

REPO = Path(__file__).resolve().parents[2]

STAGE_A_4TU = REPO / "analysis" / "processing" / "m32_stageA_lines_4tu.csv"
STAGE_A_IDS = REPO / "analysis" / "processing" / "m32_stageA_lines_ids.csv"
K_LINES = REPO / "data" / "processing" / "d3" / "lines.csv"
K_APEXES = REPO / "data" / "processing" / "d3" / "known_apexes.csv"
APEX_CANDIDATES = REPO / "docs" / "processing" / "fourtu_apex_candidates_v1.csv"
CONTENT_HEIGHT_LINES = REPO / "analysis" / "processing" / "m32_content_height_lines.csv"

EXPECTED_SHA256 = {
    "stage_a_4tu": "2d590cdab6055df22d7294ddaa11676fee39774e9b95a705b4c9884e3787cdf2",
    "stage_a_ids": "a52e467dfed80e64885b3ced9e5fb5e9e9617305b39e7d7e1d44e4d1f5bedfb3",
    "k_lines": "f6fe08a93808555727acb3ecd864abc1cd555f0c6900a753a8c598be4543ca0a",
    "k_apexes": "0fec12dba4efa504a69a8319e52eb854dac6359296beb391f6b83b177d49db61",
}
"""Plan 4-7: stage-A lists (LF bytes) and the K canon (``data/``, not in git)."""

N_STAGE_A_4TU = 150
N_STAGE_A_IDS = 60
N_K_LINES = 27
N_K_APEXES = 41

GATE_SUM_HITS = 90
"""D3 sum of ``folds_hit`` over the 41 apexes (5 folds, out of 205)."""
GATE_ANY_FOLD = 30
"""D3 apexes hit by at least one fold."""
GATE_FOLD0 = 15
"""D3 apexes hit by fold 0."""

FOLDS = (0, 1, 2, 3, 4)
SEED = 20261114
DATASET_4TU = "4tu"
DATASET_IDS = "gz_ids"
ROLE_BACKGROUND = "background"
ROLE_K = "k"

R0 = "r0"
STAGE_C_RECIPES = ("r0", "r5", "r6")
"""Plan 4-7: stage-C candidates r0 (current), r5, r6 (user decision 2026-11-14)."""

RECIPE_OPTIONS: dict[str, dict[str, object]] = {
    "r0": {},
    "r5": {
        "detection_bg_mode": "median",
        "gain_mode": "agc_capped",
        "agc_floor_beta": 0.5,
        "pad_mode": "taper",
        "pad_taper_periods": 1.0,
    },
    "r6": {
        "detection_bg_mode": "partial",
        "detection_bg_alpha": 0.8,
        "gain_mode": "agc",
        "agc_window_periods": 32.0,
        "pad_mode": "flat",
    },
}
"""Plan 4-1 table, as run in stage A (``m32_texture_recipes.py`` ``RECIPES``)."""

STAGE_A_DISTANCE = {"r0": 8.08, "r5": 5.77, "r6": 5.98}
"""``D_r`` of plan 4-7 (tie break of the selection rule)."""

BACKGROUND_REMOVAL_WEAKENED = {"r6"}
"""Recipes of the B1-B3 family (weaker background removal than B0, plan 4-8)."""


@dataclass(frozen=True)
class LineRef:
    """One line of the stage-C runs.

    Attributes:
        dataset: ``"4tu"`` or ``"gz_ids"``.
        line_id: ``"01.9/Path1"`` or ``"pipe/1.15_200115AB/LID10002"``.
        role: ``"background"`` (stage-A list) or ``"k"`` (guard M6 only).
        time_window_ns: IDS record length from the stage-A list.
    """

    dataset: str
    line_id: str
    role: str
    time_window_ns: float | None = None

    @property
    def key(self) -> str:
        """File-name-safe id."""
        return f"{self.dataset}__{self.line_id.replace('/', '__')}"


@dataclass(frozen=True)
class KnownApex:
    """One blind-review apex of the K bundle (M6)."""

    blind_id: str
    line_id: str
    trace: int
    sample: float


class InputError(Exception):
    """A missing, changed or malformed input (exit code 2)."""


def sha256_file(path: Path) -> str:
    """Hex sha256 of the file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha256(path: Path, expected: str, what: str) -> str:
    """sha256 of ``path``; ``InputError`` when missing or different."""
    if not path.is_file():
        raise InputError(f"{what} missing: {path}")
    digest = sha256_file(path)
    if digest != expected:
        raise InputError(f"{what} sha256 {digest} != expected {expected}")
    return digest


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a UTF-8 csv (BOM tolerated)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def recipe_config(recipe_id: str) -> DetectionImageConfig:
    """``DetectionImageConfig`` of a stage-C recipe (``recipe_id`` recorded).

    Raises:
        KeyError: For a recipe outside ``STAGE_C_RECIPES``.
    """
    options = RECIPE_OPTIONS[recipe_id]
    return DetectionImageConfig(**options, recipe_id=recipe_id)  # type: ignore[arg-type]


def recipes_sha256(recipe_ids: Sequence[str]) -> str:
    """sha256 of the canonical json of the recipe configs (run identity)."""
    document = {rid: asdict(recipe_config(rid)) for rid in recipe_ids}
    text = json.dumps(document, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- line lists


def assert_list_hashes() -> dict[str, str]:
    """Assert the four plan-fixed lists (stage A and the K canon)."""
    paths = {
        "stage_a_4tu": STAGE_A_4TU,
        "stage_a_ids": STAGE_A_IDS,
        "k_lines": K_LINES,
        "k_apexes": K_APEXES,
    }
    return {
        name: checked_sha256(path, EXPECTED_SHA256[name], name)
        for name, path in paths.items()
    }


def background_lines() -> list[LineRef]:
    """Stage-A lists: 4TU 150 and IDS 60 lines (plan 3-2 "배경 측선")."""
    lines = [
        LineRef(DATASET_4TU, row["line"], ROLE_BACKGROUND)
        for row in read_csv(STAGE_A_4TU)
    ]
    lines += [
        LineRef(
            DATASET_IDS,
            row["line"],
            ROLE_BACKGROUND,
            float(row["time_window_ns"]),
        )
        for row in read_csv(STAGE_A_IDS)
    ]
    counts = (
        sum(line.dataset == DATASET_4TU for line in lines),
        sum(line.dataset == DATASET_IDS for line in lines),
    )
    if counts != (N_STAGE_A_4TU, N_STAGE_A_IDS):
        raise InputError(f"stage-A lists have {counts} lines, expected 150 and 60")
    return lines


def k_lines() -> list[LineRef]:
    """The K bundle: rows of ``d3/lines.csv`` whose ``bundles`` contain K."""
    lines = [
        LineRef(DATASET_4TU, f"{row['activity_id']}/{row['path']}", ROLE_K)
        for row in read_csv(K_LINES)
        if "K" in row["bundles"].split("+")
    ]
    if len(lines) != N_K_LINES:
        raise InputError(f"K bundle has {len(lines)} lines, expected {N_K_LINES}")
    return lines


def known_apexes() -> list[KnownApex]:
    """The 41 M6 apexes of ``d3/known_apexes.csv``."""
    apexes = [
        KnownApex(row["blind_id"], row["line"], int(row["trace"]), float(row["sample"]))
        for row in read_csv(K_APEXES)
    ]
    if len(apexes) != N_K_APEXES:
        raise InputError(f"known_apexes has {len(apexes)} rows, expected 41")
    return apexes


def uninjected_lines() -> list[LineRef]:
    """P5a lines: stage-A backgrounds plus the K bundle (checked disjoint).

    Raises:
        InputError: If a K line is a background or the apex lines differ
            from the K rows (plan 4-7 "K 묶음 정본").
    """
    backgrounds = background_lines()
    k_bundle = k_lines()
    background_ids = {line.line_id for line in backgrounds}
    overlap = background_ids & {line.line_id for line in k_bundle}
    if overlap:
        raise InputError(f"K lines inside the background lists: {sorted(overlap)}")
    apex_lines = {apex.line_id for apex in known_apexes()}
    if apex_lines != {line.line_id for line in k_bundle}:
        raise InputError("known_apexes lines differ from the K rows of d3/lines.csv")
    return backgrounds + k_bundle


@dataclass(frozen=True)
class CandidateApexes:
    """Zone (d) inputs per line, kept apart since amendment 5 ([CVR] C-2).

    Attributes:
        k: The 41 K apexes (``known_apexes.csv``, sha asserted); radius -20 dB.
        auto: ``fourtu_apex_candidates_v1.csv`` apexes; radius -6 dB + 1 P.
    """

    k: dict[str, tuple[float, ...]]
    auto: dict[str, tuple[float, ...]]

    def k_traces(self, line_id: str) -> tuple[float, ...]:
        """K apex traces of one line."""
        return self.k.get(line_id, ())

    def auto_traces(self, line_id: str) -> tuple[float, ...]:
        """Automatic candidate traces of one line."""
        return self.auto.get(line_id, ())


def _sorted_by_line(pairs: Iterable[tuple[str, float]]) -> dict[str, tuple[float, ...]]:
    traces: dict[str, list[float]] = {}
    for line_id, trace in pairs:
        traces.setdefault(line_id, []).append(trace)
    return {line_id: tuple(sorted(values)) for line_id, values in traces.items()}


def candidate_apexes() -> CandidateApexes:
    """K apexes and automatic candidates per line, as two separate sets."""
    auto = (
        (f"{row['activity_id']}/{row['path']}", float(row["trace"]))
        for row in read_csv(APEX_CANDIDATES)
    )
    k = ((apex.line_id, float(apex.trace)) for apex in known_apexes())
    return CandidateApexes(k=_sorted_by_line(k), auto=_sorted_by_line(auto))


# --------------------------------------------------------------- exclusions


EXCLUSION_COLUMNS = ("line_id", "i0", "i1", "k0", "k1")


def write_exclusions(path: Path, boxes: Mapping[str, Sequence[GridBox]]) -> str:
    """Zone (c') boxes per line (sorted, unique) as LF csv; returns the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(EXCLUSION_COLUMNS)
        for line_id in sorted(boxes):
            unique = sorted({(b.i0, b.i1, b.k0, b.k1) for b in boxes[line_id]})
            for i0, i1, k0, k1 in unique:
                writer.writerow([line_id, i0, i1, k0, k1])
    return sha256_file(path)


def read_exclusions(path: Path) -> dict[str, tuple[GridBox, ...]]:
    """Inverse of :func:`write_exclusions`."""
    boxes: dict[str, list[GridBox]] = {}
    for row in read_csv(path):
        box = GridBox(int(row["i0"]), int(row["i1"]), int(row["k0"]), int(row["k1"]))
        boxes.setdefault(row["line_id"], []).append(box)
    return {line_id: tuple(values) for line_id, values in boxes.items()}
