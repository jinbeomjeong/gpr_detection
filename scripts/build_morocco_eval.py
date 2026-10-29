"""Build the Morocco cross-domain evaluation set: one near-original image per group.

Checklist M1-4. The Morocco package is read in place (never written).

Outputs (``--out`` defaults to ``data/morocco_eval_v1``):

* ``<out>/images/*.jpg`` - original JPG bytes at their original size
* ``<out>/labels/*.txt`` - YOLO labels, classes 0 pipe_hyperbola and 2 cavity;
  empty file = intact (background) image
* ``<out>/eval_images.txt`` + ``<out>/eval.yaml`` - Ultralytics config with
  val = test = the evaluation list (evaluation only, never train on it)
* ``<out>/summary.json`` - groups, images, boxes, fallbacks and sizes per subset
* ``data/splits/morocco_eval_v1.csv`` - manifest (``split_src``,
  ``anisotropic_stretch``, ``selection_rule`` per image)

Examples:
    conda run -n gpr-dev python -I scripts/build_morocco_eval.py --dry-run
    conda run -n gpr-dev python -I scripts/build_morocco_eval.py
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from gpr_detection.data.morocco_eval import MoroccoEvalConfig, build_morocco_eval
from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "morocco_eval_v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "morocco_eval_v1.csv"


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
        default=DEFAULT_MANIFEST,
        help="manifest CSV path",
    )
    parser.add_argument(
        "--morocco-root",
        type=Path,
        default=None,
        help="Morocco package root (default: configs/paths.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and read labels only, print the counts; write nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build and print the summary as JSON.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = MoroccoEvalConfig(
        morocco_root=args.morocco_root or load_dataset_paths().morocco,
        out_dir=args.out,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )

    started_s = time.perf_counter()
    summary = build_morocco_eval(config)
    elapsed_s = time.perf_counter() - started_s

    # ASCII JSON keeps the console output readable under cp949.
    print(json.dumps(summary, indent=2))
    action = "planned (dry run)" if config.dry_run else f"written to {config.out_dir}"
    print(
        f"{summary['total']['images']} images {action} in {elapsed_s:.1f} s",
        file=sys.stderr,
    )
    if not config.dry_run:
        print(f"manifest: {config.manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
