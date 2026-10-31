"""Band pick quality on AIHub train_pool ``pipe_band`` boxes (M2-6b, band spec 9-4).

No-scale check only: AIHub crops have no dt or dx, so every result must be
``no_scale`` with all m and ns fields ``None`` (QA-4), and the picks should
still come out. No depth is judged (there is no depth truth).

Data: ``data/splits/aihub_yolo_v1.csv`` rows with ``subset == "trainpool"``
(folds 0-4; the test split stays sealed until M4-3), gain 1.0, containing class
1 (``pipe_band``); images and YOLO labels from ``data/aihub_yolo_v1``. A
seeded sample of ``--count`` images is used.

Outputs: aggregate markdown ``docs/processing/<stem>.md`` (no file names or box
coordinates; AIHub data card rule) and a per-box csv under ``data/audit/``
(local, not tracked). Only the block between the ``GENERATED_BEGIN`` and
``GENERATED_END`` markers is rewritten; hand-written notes outside it are kept.
An existing report without the markers is left alone (exit 2). The version
table puts the current run in a column named after ``CURRENT_SPEC_VERSION``
next to the frozen ``PREVIOUS_RUNS`` columns.

Example:
    conda run -n gpr-dev python -I scripts/check_band_picks_aihub.py --count 40
"""

import argparse
import csv
import logging
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gpr_detection.depth.band import band_exclude_traces, estimate_band_depths
from gpr_detection.depth.result import DepthResult
from gpr_detection.io._common import read_grayscale_file
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.preprocess.filters import (
    dominant_period_px,
    prepare_fitting_branch,
    remove_dc,
)
from gpr_detection.preprocess.timezero import find_time_zero

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
DATASET_DIR = PROJECT_ROOT / "data" / "aihub_yolo_v1"
DEFAULT_MD_DIR = PROJECT_ROOT / "docs" / "processing"
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "audit"
DEFAULT_STEM = "depth_band_aihub_trainpool_v1"

PIPE_BAND_CLASS = 1
TRAIN_FOLDS = {"0", "1", "2", "3", "4"}
GAIN = "1.0"
SEED = 0

SCALED_RESULT_FIELDS = (
    "depth_top_m",
    "depth_center_m",
    "sigma_depth_m",
    "v_m_per_ns",
    "sigma_v",
    "eps_r_used",
    "t_zero_ns",
    "t0_ns",
    "t_apex_ns",
    "x0_m",
)
SCALED_BAND_FIELDS = (
    "x_ref_m",
    "t_band_ns",
    "slope_ns_per_m",
    "grade",
    "depth_profile",
    "lateral_bias_max_m",
)

LOGGER = logging.getLogger("check_band_picks_aihub")


def sample_images(count: int) -> list[str]:
    """Seeded sample of train_pool gain-1.0 crops that contain ``pipe_band``."""
    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    has_band = (
        split["classes"].str.split(";").apply(lambda c: str(PIPE_BAND_CLASS) in c)
    )
    chosen = split[
        (split["subset"] == "trainpool")
        & split["fold"].isin(TRAIN_FOLDS)
        & (split["gain"] == GAIN)
        & has_band
    ]
    names = sorted(chosen["file_name"])
    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(names), size=min(count, len(names)), replace=False)
    return [names[i] for i in sorted(picked)]


def band_boxes(label_path: Path, width: int, height: int) -> list[GridBox]:
    """``pipe_band`` YOLO boxes -> L0 boxes (pixel j covers ``[j, j + 1)``)."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or int(parts[0]) != PIPE_BAND_CLASS:
            continue
        xc, yc, w, h = (float(value) for value in parts[1:5])
        i0 = max(0, math.floor((xc - w / 2) * width))
        i1 = min(width - 1, math.ceil((xc + w / 2) * width) - 1)
        k0 = max(0, math.floor((yc - h / 2) * height))
        k1 = min(height - 1, math.ceil((yc + h / 2) * height) - 1)
        if i1 >= i0 and k1 >= k0:
            boxes.append(GridBox(i0=i0, i1=i1, k0=k0, k1=k1))
    return boxes


@dataclass(frozen=True)
class ImageResult:
    """One crop: its band boxes, results and dominant period."""

    boxes: list[GridBox]
    results: list[DepthResult]
    period_px: float | None
    """``bandpass_px`` period; ``None`` when skipped (outside 2.2-128 px)."""
    dominant_period_px: float | None
    """``1 / f_px`` without the range check (reported for skipped crops)."""


def evaluate_image(file_name: str) -> ImageResult:
    image = read_grayscale_file(DATASET_DIR / "images" / file_name)
    height, width = image.shape
    boxes = band_boxes(
        DATASET_DIR / "labels" / f"{Path(file_name).stem}.txt", width, height
    )
    radargram = Radargram(
        data=image.astype(np.float32),
        dt_ns=None,
        dx_m=None,
        air_gap="unknown",
        amplitude_processed=False,
        dataset="aihub",
        line_id=Path(file_name).stem,
    )
    branch = prepare_fitting_branch(radargram)
    excluded = band_exclude_traces(boxes, width, None)
    zero = find_time_zero(radargram, branch.filtered, exclude_traces=excluded)
    results = estimate_band_depths(radargram, boxes, zero, branch=branch)
    dominant = dominant_period_px(remove_dc(radargram.data))
    return ImageResult(boxes, results, branch.period_px, dominant)


def scaled_fields_present(result: DepthResult) -> list[str]:
    """Names of m/ns fields that are not ``None`` (QA-4 violations)."""
    present = [
        name for name in SCALED_RESULT_FIELDS if getattr(result, name) is not None
    ]
    if result.band is not None:
        present += [
            f"band.{name}"
            for name in SCALED_BAND_FIELDS
            if getattr(result.band, name) is not None
        ]
    return present


def result_row(
    boxes: Sequence[GridBox], result: DepthResult, period_px: float | None
) -> dict[str, object]:
    band = result.band
    members = [] if band is None else list(band.member_indices)
    top_k0 = min((boxes[i].k0 for i in members), default=None)
    res_over_period = (
        ""
        if band is None or period_px is None
        else round(band.sigma_res / period_px, 3)
    )
    return {
        "status": result.status.value,
        "has_picks": band is not None and band.n_picks > 0,
        "n_members": len(members),
        "model": "" if band is None else band.model,
        "coverage": "" if band is None else round(band.coverage, 3),
        "confidence": "" if band is None else round(band.confidence, 3),
        "sigma_res_px": "" if band is None else round(band.sigma_res, 3),
        "period_px": "" if period_px is None else round(period_px, 3),
        "sigma_res_over_period": res_over_period,
        "k_ref_minus_k0_px": (
            "" if band is None or top_k0 is None else round(band.ref_sample - top_k0, 2)
        ),
        "scaled_fields_present": " ".join(scaled_fields_present(result)),
        "flags": " ".join(result.flags),
    }


def _quantiles(values: Sequence[float]) -> str:
    if not values:
        return "-"
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return " / ".join(f"{value:.2f}" for value in q)


CURRENT_SPEC_VERSION = "v1.4"
"""Band spec version the current code implements.

Taken from the change history of ``docs/processing/spec_band_depth_v1.md``;
bump it (and move the old current column into ``PREVIOUS_RUNS``) when the band
spec changes what this check measures.
"""
CURRENT_RUN_SUMMARY = "S1 겹침 판정 + G 32 px"
"""What the current spec version changed, shown in the current column header."""

PREVIOUS_RUNS = {
    "v1.0 (고정 1/2/2/3/1 px)": {
        "inconsistent_picks": "30/51",
        "band_concave": "19/51",
        "confidence": "0.00 / 0.00 / 0.00 / 0.00 / 0.40",
        "skipped": "16/51 결과 (영상 수 미기록)",
        "sigma_res_over_period": "-",
        "results_and_fragments": "-",
    },
    "v1.1 (P_px 기준, 하한 4 px)": {
        "inconsistent_picks": "39/51",
        "band_concave": "21/51",
        "confidence": "0.00 / 0.00 / 0.00 / 0.00 / 0.29",
        "skipped": "16/51 결과, 14장 (1/f_px 2.44-3.82 px)",
        "sigma_res_over_period": "-",
        "results_and_fragments": "-",
    },
    "v1.3 (P_px 기준, 하한 2.2 px)": {
        "inconsistent_picks": "42/51",
        "band_concave": "20/51",
        "confidence": "0.00 / 0.00 / 0.00 / 0.00 / 0.29",
        "skipped": "0/51 결과, 0장",
        "sigma_res_over_period": "0.08 / 0.35 / 0.52 / 0.84 / 1.52",
        "results_and_fragments": "51 / 6",
    },
}
"""Frozen earlier runs on the same 40 crops (seed 0), kept for the comparison table.

v1.0 and v1.1 ran on 2026-10-30; v1.3 is the run published in the report before
the v1.4 rerun (2026-10-31). Values are copied from the report, not recomputed.
"""

COMPARISON_ROW_LABELS = {
    "inconsistent_picks": "flag `inconsistent_picks`",
    "band_concave": "flag `band_concave`",
    "confidence": "confidence 분위",
    "skipped": "`bandpass_px_skipped`",
    "sigma_res_over_period": "sigma_res / P_px 분위",
    "results_and_fragments": "박스 {n_boxes}개 → 결과 / `fragment_of_group`",
}

GENERATED_BEGIN = "<!-- generated: scripts/check_band_picks_aihub.py begin -->"
GENERATED_END = (
    "<!-- generated: scripts/check_band_picks_aihub.py end"
    " (hand-written notes go below this line) -->"
)


def current_run_label() -> str:
    """Comparison-table header of the run made by this code."""
    return f"{CURRENT_SPEC_VERSION} ({CURRENT_RUN_SUMMARY}, 이번)"


def current_run_column(
    rows: Sequence[dict[str, object]], images: Sequence[ImageResult]
) -> dict[str, str]:
    """Comparison-table cells of this run, keyed like ``PREVIOUS_RUNS`` entries."""
    flags = Counter(flag for row in rows for flag in str(row["flags"]).split() if flag)
    total = len(rows)
    skipped_images = sum(image.period_px is None for image in images)
    confidence = [float(row["confidence"]) for row in rows if row["confidence"] != ""]
    ratios = [
        float(row["sigma_res_over_period"])
        for row in rows
        if row["sigma_res_over_period"] != ""
    ]
    return {
        "inconsistent_picks": f"{flags['inconsistent_picks']}/{total}",
        "band_concave": f"{flags['band_concave']}/{total}",
        "confidence": _quantiles(confidence),
        "skipped": f"{flags['bandpass_px_skipped']}/{total} 결과, {skipped_images}장",
        "sigma_res_over_period": _quantiles(ratios),
        "results_and_fragments": f"{total} / {flags['fragment_of_group']}",
    }


def comparison_table(current: dict[str, str], n_boxes: int) -> list[str]:
    """Markdown table: frozen ``PREVIOUS_RUNS`` columns, then the current run."""
    columns = {**PREVIOUS_RUNS, current_run_label(): current}
    lines = [
        "| 항목 | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for key, label in COMPARISON_ROW_LABELS.items():
        cells = " | ".join(run[key] for run in columns.values())
        lines.append(f"| {label.format(n_boxes=n_boxes)} | {cells} |")
    return lines


def merge_generated_block(existing: str | None, generated: str) -> str:
    """Put ``generated`` between the markers, keeping text outside them.

    Args:
        existing: Current report text, or ``None`` when the report is new.
        generated: Report body without markers.

    Returns:
        The full report text.

    Raises:
        ValueError: The existing report lacks exactly one ordered marker pair;
            overwriting it could destroy hand-written notes.
    """
    block = f"{GENERATED_BEGIN}\n{generated}\n{GENERATED_END}"
    if existing is None:
        return f"{block}\n"
    if existing.count(GENERATED_BEGIN) != 1 or existing.count(GENERATED_END) != 1:
        raise ValueError("report needs exactly one generated begin and end marker")
    begin = existing.index(GENERATED_BEGIN)
    end = existing.index(GENERATED_END)
    if end < begin:
        raise ValueError("generated end marker comes before the begin marker")
    return existing[:begin] + block + existing[end + len(GENERATED_END) :]


def _period_lines(images: Sequence[ImageResult]) -> list[str]:
    applied = [image.period_px for image in images if image.period_px is not None]
    skipped = [
        image.dominant_period_px
        for image in images
        if image.period_px is None and image.dominant_period_px is not None
    ]
    return [
        f"| P_px 적용 영상 수 / 분위(px) | {len(applied)} / {_quantiles(applied)} |",
        f"| `bandpass_px_skipped` 영상 수 / 1/f_px 분위(px) | {len(skipped)} / "
        f"{_quantiles(skipped)} |",
    ]


def write_markdown(
    path: Path,
    images: Sequence[ImageResult],
    rows: Sequence[dict[str, object]],
) -> None:
    """Rewrite the generated block of the report; text outside it is kept.

    Raises:
        ValueError: The report exists but has no generated markers.
    """
    n_images = len(images)
    n_boxes = sum(len(image.boxes) for image in images)
    statuses = Counter(row["status"] for row in rows)
    flags = Counter(flag for row in rows for flag in str(row["flags"]).split() if flag)
    with_picks = sum(bool(row["has_picks"]) for row in rows)
    violations = [row for row in rows if row["scaled_fields_present"]]
    numbers = {
        key: [float(row[key]) for row in rows if row[key] != ""]
        for key in (
            "coverage",
            "confidence",
            "k_ref_minus_k0_px",
            "sigma_res_over_period",
        )
    }
    lines = [
        "# AIHub train_pool `pipe_band` 픽 품질 점검 v1 (M2-6b, band spec §9-4)",
        "",
        "- 대상: `data/splits/aihub_yolo_v1.csv` trainpool(fold 0-4), gain 1.0, "
        f"`pipe_band` 포함 crop에서 시드 {SEED} 표본 {n_images}장, 박스 {n_boxes}개 "
        "(test 분할은 M4-3까지 봉인이라 쓰지 않음)",
        "- 스케일 없음(dt·dx 없음): 판정 없음. "
        "파일명·박스 좌표는 싣지 않는다(AIHub 카드 규칙)",
        f"- 조각 묶기: 박스 {n_boxes}개 → 결과 {len(rows)}개",
        f"- 픽이 나온 결과: {with_picks}/{len(rows)}",
        f"- QA-4 m·ns 필드가 채워진 결과: {len(violations)}개",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        *[f"| status `{name}` | {count} |" for name, count in statuses.most_common()],
        f"| coverage 분위(최소/25/50/75/최대) | {_quantiles(numbers['coverage'])} |",
        f"| confidence 분위 | {_quantiles(numbers['confidence'])} |",
        f"| k_ref - k0 (px) 분위 | {_quantiles(numbers['k_ref_minus_k0_px'])} |",
        f"| sigma_res / P_px 분위 (문턱 0.25) | "
        f"{_quantiles(numbers['sigma_res_over_period'])} |",
        *[f"| flag `{name}` | {count} |" for name, count in flags.most_common()],
        *_period_lines(images),
        "",
        "스케일 없는 입력의 px 기준은 band spec §3-5대로 우세 주기 P_px에 "
        "묶었다(f_scale 0.05P, 잔차 문턱 0.25P, sloped 0.1P, 묶기 max(0.5P, 3), "
        "q_res 0.125P; `bandpass_px_skipped`면 고정 1/2/2/3/1 px). "
        "P_px 범위 하한은 [S] §3 v1.7의 2.2 px다.",
        "",
        "## 버전 비교 (같은 40장, 시드 0)",
        "",
        *comparison_table(current_run_column(rows, images), n_boxes),
    ]
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    path.write_text(merge_generated_block(existing, "\n".join(lines)), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check; exit 1 on a QA-4 violation, 2 when the report has no markers."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    args = parser.parse_args(argv)

    names = sample_images(args.count)
    images = [evaluate_image(name) for name in names]
    rows = [
        result_row(image.boxes, result, image.period_px)
        for image in images
        for result in image.results
    ]
    n_boxes = sum(len(image.boxes) for image in images)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    with (args.csv_dir / f"{args.stem}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.md_dir / f"{args.stem}.md"
    try:
        write_markdown(report_path, images, rows)
    except ValueError as error:
        LOGGER.error("%s not rewritten: %s", report_path, error)
        return 2
    violations = sum(bool(row["scaled_fields_present"]) for row in rows)
    LOGGER.info(
        "images %d, boxes %d, results %d, QA-4 violations %d",
        len(names),
        n_boxes,
        len(rows),
        violations,
    )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
