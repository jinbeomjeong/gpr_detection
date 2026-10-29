"""Command-line entry point ``gpr-detect`` (stub until M5-1 adds ``infer``)."""

import argparse
from collections.abc import Sequence

from gpr_detection import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpr-detect",
        description="Detect buried pipes in GPR B-scans and estimate their depth.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
