"""Build the auto-merged pipe_band dataset v2 from data/aihub_yolo_v1 (no relabel).

User decision 2026-10-31: merge fragmented ``pipe_band`` boxes automatically
with gap <= 32 px (``labeling/pipe_band_relabel_plan.md`` section 8), apply the
same rule to test and keep the v1 test labels as ``<out>/test_v1``. v1 is only
read. After writing, the leakage re-assertions (1)-(6) re-read the outputs.

Outputs (``--out`` defaults to ``data/aihub_yolo_v2_auto<gap>``): see
:mod:`gpr_detection.data.aihub_yolo_v2_merge`; manifest
``data/splits/<out name>.csv``.

The last stdout line is one JSON object with the counts and the check results.
Exit codes: 0 written and every check passed, 1 a check failed, 2 input error
(missing input, output already present, gain copies merging differently).

Examples:
    conda run -n gpr-dev python -I scripts/build_yolo_v2_merged.py --dry-run
    conda run -n gpr-dev python -I scripts/build_yolo_v2_merged.py \\
        --merge-band-gap 32
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpr_detection.data.aihub_yolo_v2_checks import CheckInputs, run_all_checks
from gpr_detection.data.aihub_yolo_v2_merge import (
    MergeBuildConfig,
    build_merged_dataset,
)
from gpr_detection.data.band_merge import (
    DEFAULT_MAX_HEIGHT_RATIO,
    DEFAULT_MIN_Y_OVERLAP_FRACTION,
    MergeRule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_SOURCE_DIR = DATA_DIR / "aihub_yolo_v1"
DEFAULT_SOURCE_MANIFEST = SPLITS_DIR / "aihub_yolo_v1.csv"
DEFAULT_GAP_PX = 32.0

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_INPUT_ERROR = 2

LOGGER = logging.getLogger("build_yolo_v2_merged")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: data/aihub_yolo_v2_auto<gap>)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="default: data/splits/<out>.csv"
    )
    parser.add_argument(
        "--merge-band-gap",
        type=float,
        default=DEFAULT_GAP_PX,
        help="largest x gap in pixels of the original crop that is bridged",
    )
    parser.add_argument(
        "--min-y-overlap",
        type=float,
        default=DEFAULT_MIN_Y_OVERLAP_FRACTION,
        help="least y overlap as a fraction of the smaller box height",
    )
    parser.add_argument(
        "--max-height-ratio",
        type=float,
        default=DEFAULT_MAX_HEIGHT_RATIO,
        help="largest union height / taller box height (plan section 8-3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="merge in memory, print counts only"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MergeBuildConfig:
    out_dir = args.out or DATA_DIR / f"aihub_yolo_v2_auto{args.merge_band_gap:g}"
    manifest_path = args.manifest or SPLITS_DIR / f"{out_dir.name}.csv"
    rule = MergeRule(
        max_gap_px=args.merge_band_gap,
        min_y_overlap_fraction=args.min_y_overlap,
        max_height_ratio=args.max_height_ratio,
    )
    return MergeBuildConfig(
        source_dir=args.source,
        source_manifest=args.source_manifest,
        out_dir=out_dir,
        manifest_path=manifest_path,
        rule=rule,
        dry_run=args.dry_run,
    )


def _short_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("band_boxes_v1", "band_boxes_v2", "images_changed", "merged_pairs")
    return {
        level: {key: summary[level][key] for key in keys}
        for level in ("image_level", "crop_level")
    }


def _run(config: MergeBuildConfig) -> int:
    started_s = time.perf_counter()
    summary = build_merged_dataset(config)
    LOGGER.info("merged in %.1f s", time.perf_counter() - started_s)
    result: dict[str, Any] = {"dry_run": config.dry_run, **_short_summary(summary)}
    if config.dry_run:
        print(json.dumps(summary, indent=2))
        print(json.dumps(result))
        return EXIT_OK

    inputs = CheckInputs(
        v1_dir=config.source_dir,
        v1_manifest=config.source_manifest,
        v2_dir=config.out_dir,
        v2_manifest=config.manifest_path,
    )
    checks = run_all_checks(inputs)
    checks_path = config.out_dir / "leak_checks.json"
    checks_text = json.dumps([asdict(c) for c in checks], ensure_ascii=False, indent=2)
    checks_path.write_text(checks_text, encoding="utf-8")
    for check in checks:
        LOGGER.info("%s: %s", check.name, "pass" if check.passed else "FAIL")
    result["checks"] = {check.name: check.passed for check in checks}
    result["out_dir"] = str(config.out_dir)
    result["manifest"] = str(config.manifest_path)
    print(json.dumps(result))
    return EXIT_OK if all(check.passed for check in checks) else EXIT_CHECK_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Build the v2 dataset and run the re-assertions.

    Returns:
        Process exit code: 0 done, 1 a check failed, 2 input error.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = _config_from_args(_build_parser().parse_args(argv))
    try:
        return _run(config)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    sys.exit(main())
