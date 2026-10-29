"""Build the AIHub 71327 B-scan YOLO dataset: held-out test + session-grouped k-fold.

Checklist M1-3. The AIHub zips are read in place (never extracted or written).

Outputs (``--out`` defaults to ``data/aihub_yolo_v1``):

* ``<out>/images/*.jpg`` - original JPG bytes, unchanged
* ``<out>/labels/*.txt`` - YOLO ``class xc yc w h`` (normalised, classes 0-3)
* ``<out>/fold{k}_images.txt`` + ``<out>/fold{k}.yaml`` - Ultralytics configs:
  train = the other train-pool folds, val = fold k
* ``<out>/test_images.txt`` + ``<out>/test.yaml`` - train = whole train pool,
  val = test = held-out test sessions
* ``<out>/summary.json`` - subset, fold, class, session and downsampling counts
* ``data/splits/<out name>.csv`` - manifest with subset, fold and seed per image

Examples:
    conda run -n gpr-dev python -I scripts/build_yolo_dataset.py --dry-run
    conda run -n gpr-dev python -I scripts/build_yolo_dataset.py --seed 20261029
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from gpr_detection.data.aihub_yolo import SplitOptions
from gpr_detection.data.aihub_yolo_writer import BuildConfig, build_dataset
from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "aihub_yolo_v1"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_SEED = 20261029
DEFAULT_OPTIONS = SplitOptions(seed=DEFAULT_SEED)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output dataset folder (must be empty or absent)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest CSV (default: data/splits/<out name>.csv)",
    )
    parser.add_argument(
        "--aihub-root",
        type=Path,
        default=None,
        help="AIHub 01-1.정식개방데이터 folder (default: configs/paths.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed of the test draw, the downsampling draw and the fold split",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_OPTIONS.k, help="number of train-pool folds"
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=DEFAULT_OPTIONS.balance_ratio,
        help="cavity and manhole images each = ratio x pipe images (per subset)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_OPTIONS.test_fraction,
        help="share of session groups held out as the test subset",
    )
    parser.add_argument(
        "--min-test-pipe-sessions",
        type=int,
        default=DEFAULT_OPTIONS.min_test_pipe_sessions,
        help="minimum tpipe and lpipe sessions each in the test subset",
    )
    parser.add_argument(
        "--pipe-share-min",
        type=float,
        default=DEFAULT_OPTIONS.pipe_share_min,
        help="lowest accepted test share of tpipe images and of lpipe images",
    )
    parser.add_argument(
        "--pipe-share-max",
        type=float,
        default=DEFAULT_OPTIONS.pipe_share_max,
        help="highest accepted test share of tpipe images and of lpipe images",
    )
    parser.add_argument(
        "--max-test-attempts",
        type=int,
        default=DEFAULT_OPTIONS.max_test_attempts,
        help="test draws tried (streams '<seed>:test:<attempt>') before failing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="read only the first N labels per zip (for tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read labels only and print the counts; write nothing",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildConfig:
    aihub_root = args.aihub_root or load_dataset_paths().aihub
    manifest_path = args.manifest or SPLITS_DIR / f"{args.out.name}.csv"
    options = SplitOptions(
        seed=args.seed,
        k=args.k,
        balance_ratio=args.balance_ratio,
        test_fraction=args.test_fraction,
        min_test_pipe_sessions=args.min_test_pipe_sessions,
        pipe_share_min=args.pipe_share_min,
        pipe_share_max=args.pipe_share_max,
        max_test_attempts=args.max_test_attempts,
    )
    return BuildConfig(
        aihub_root=aihub_root,
        out_dir=args.out,
        manifest_path=manifest_path,
        options=options,
        limit_per_zip=args.limit,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build and print the summary as JSON.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = _config_from_args(args)

    started_s = time.perf_counter()
    summary = build_dataset(config)
    elapsed_s = time.perf_counter() - started_s

    # ASCII JSON keeps the console output readable under cp949.
    print(json.dumps(summary, indent=2))
    action = "planned (dry run)" if config.dry_run else f"written to {config.out_dir}"
    print(
        f"{summary['selected']['images']} images {action} in {elapsed_s:.1f} s",
        file=sys.stderr,
    )
    if not config.dry_run:
        print(f"manifest: {config.manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
