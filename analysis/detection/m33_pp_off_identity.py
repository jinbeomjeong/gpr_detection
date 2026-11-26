"""M3-3 PP spec §5 1b: PP-off line detections of one ``src`` tree on the r0 11 lines.

Run it once with the HEAD a0dc987 tree (before PP, ``git archive a0dc987 src``)
and once with the working tree; equal ``digest`` values mean ``detect_line``
with PP off gives the same merged line boxes and L0 boxes bit for bit. PP off
is named (``pp_mode=PPSetId.S0``) whenever ``detect_line`` has ``pp_mode``,
since the M3-3 close made S3 the default.

The lines are the [SP] §9 candidates (``tests/test_detection_image_data.py``)
read through r0 (``DEFAULT_DETECTION_IMAGE_CONFIG``); the detector is the
deterministic stand-in of ``tests/test_pp.py`` (no weights: no detection
result of the judged domains is produced before C1).

Usage::

    python -I -B m33_pp_off_identity.py --src <tree>/src --out <file>.json

Stdout: the imported ``gpr_detection`` path, one line per radargram (name,
box count, digest) and the overall sha256 as the last line.
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="src folder to use")
    parser.add_argument("--out", type=Path, required=True, help="JSON digest file")
    return parser.parse_args(argv)


def _box_record(line_box, grid_box) -> list:
    """Every field of a merged box (``repr`` keeps float bits) and its L0 box."""
    return [repr(line_box), [grid_box.i0, grid_box.i1, grid_box.k0, grid_box.k1]]


def line_digest(detections) -> tuple[int, str]:
    """Box count and sha256 of the merged and L0 boxes of one line."""
    records = [_box_record(box, grid) for box, grid in detections.boxes]
    payload = json.dumps([records, detections.tile_seam_box], sort_keys=True)
    return len(records), hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pp_off_options(detect_line) -> dict:
    """``pp_mode=S0`` when ``detect_line`` has it, else nothing (a0dc987 tree).

    Since the M3-3 close the default is the C0b set S3, so PP off must be
    named; the tree before PP has no ``pp_mode`` and is always off.
    """
    if "pp_mode" not in inspect.signature(detect_line).parameters:
        return {}
    from gpr_detection.pipeline import PPSetId

    return {"pp_mode": PPSetId.S0}


def main(argv: list[str] | None = None) -> int:
    """Digest the 11 lines with the given ``src`` tree; returns 0."""
    args = _parse_args(argv)
    # The chosen tree must win over the editable install of the env.
    sys.path.insert(0, str(args.src.resolve()))
    sys.path.insert(1, str(TESTS))
    import gpr_detection
    import test_detection_image_data as candidates
    from fake_line_detector import TileStatisticsDetector
    from gpr_detection.io.paths import load_dataset_paths
    from gpr_detection.pipeline import detect_line

    print(f"gpr_detection: {Path(gpr_detection.__file__).resolve()}", flush=True)
    paths = load_dataset_paths()
    pp_off = pp_off_options(detect_line)
    lines = {}
    for candidate in candidates.CANDIDATES:
        radargram = candidate.load(paths.root(candidate.dataset))
        detections = detect_line(radargram, TileStatisticsDetector(), **pp_off)
        n_boxes, digest = line_digest(detections)
        lines[candidate.name] = {"n_boxes": n_boxes, "digest": digest}
        print(f"{candidate.name} {n_boxes} {digest}", flush=True)
    overall = hashlib.sha256(
        json.dumps(lines, sort_keys=True).encode("utf-8")
    ).hexdigest()
    document = {
        "src": str(args.src.resolve()),
        "n_lines": len(lines),
        "lines": lines,
        "sha256": overall,
    }
    args.out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(overall, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
