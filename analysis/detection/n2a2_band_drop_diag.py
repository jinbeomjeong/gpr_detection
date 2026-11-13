"""Why the N2-a' context step drops labels (report only, CPU, no model).

The A1 judgement (docs/experiments/m31_n2a2_a1_judgment.md) reports that the
``context_scale`` stage dropped about 43 % of the pipe_band labels it saw,
against 1-4 % for the N2-a scale stage. That training counter mixes several
causes. This script pushes N fold-0 training crops through the sealed stage
(``gpr_detection.augment_geometry.augment_context_scale`` at commit fa4b91c,
config ``configs/train_yolo11m_n2a2.yaml``) once with the context mode forced
and once with the plain mode forced, same images and seed, and splits every
lost label by cause:

  merge stage (label merge of the strip, section 3-2):
    (v) ``duplicate``: removed by the same-class duplicate rules (가)/(나);
        ``replaced_by_larger`` counts centre boxes replaced by rule (가)
        (the object keeps a label, only the box changes);
    ``neighbour_in_centre_columns`` / ``voronoi_other_neighbour`` /
    ``band_no_owned_columns``: neighbour labels handed to another crop
    (ownership, not a loss);
  window stage (keep rule after scale + 256 px window, section 2-2 step 5):
    (i)   ``outside_window_neighbour``: a neighbour label left wholly outside;
    (ii)  ``outside_window_centre``: a centre-crop label pushed wholly outside;
    (iii) ``below_half_visible_{centre,neighbour}``: partly visible, visible
          area < 50 % of the scaled box (with the band length distribution);
    (iv)  ``min_side``: visible share >= 50 % but a cut side < 4 px.

The training counter (``aug_stats`` stage ``context_scale``) is the window
stage only: dropped / (kept + dropped) over the merged labels. The same share
is reported here per mode and for an even mix of the two modes (training uses
p_context 0.5), so the two can be compared.

Image-level shares: images with >= 1 band label dropped, and images whose
centre crop has a band label but none of the centre's band labels (or no band
label at all) survives the step; the latter are band images that train as
background for this step.

Instrumentation only: ``merge_strip_labels`` and ``place_window`` are wrapped
at run time to record their inputs and outputs; the sealed source is not
changed (the script checks that ``src/`` equals the seal commit). Ultralytics
steps after this one (translate, flip, shift / edge fill) are not included.
Centre labels are read with the same parser as neighbour labels (float64 from
the txt, not the float32 Ultralytics cache).

stdout carries only the last line {"status", "exit_code", "out"}. --out must not
exist unless --overwrite. Exit codes: 0 written, 2 input missing / --out
exists / source differs from the seal, 3 internal error.

Example (PowerShell, pinned, BelowNormal):
    conda run -n gpr-dev python -I analysis/detection/n2a2_band_drop_diag.py \
        --images 2000 --seed 0
"""

import argparse
import dataclasses
import itertools
import logging
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

from gpr_detection import augment_geometry
from gpr_detection.augment_geometry import (
    ContextIndex,
    CropContent,
    StripPiece,
    augment_context_scale,
    box_areas,
    context_config_from_dict,
    geometric_config_from_dict,
)
from gpr_detection.detect.yolo_train import FolderCropLoader

REPO = common.REPO
SEAL_COMMIT = "fa4b91c"
CONFIG = REPO / "configs" / "train_yolo11m_n2a2.yaml"
DATASET_DIR = REPO / "data" / "aihub_yolo_v1"
DEFAULT_OUT = Path(__file__).resolve().parent / "n2a2_band_drop_diag.json"
SCHEMA = "n2a2_band_drop_diag/1"
CLASS_NAMES = common.CLASS_NAMES
BAND = CLASS_NAMES.index("pipe_band")
BAND_LENGTH_EDGES_PX = (0, 32, 64, 128, 192, 256, 320, 384, 448, 512)
VISIBLE_SHARE_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CENTRE, NEIGHBOUR = "centre", "neighbour"
GIT_TIMEOUT_S = 30

logger = logging.getLogger("n2a2_band_drop_diag")


# ------------------------------------------------------ instrumentation


@dataclass
class _Capture:
    """Inputs and outputs of the wrapped functions for one image."""

    merge_pieces: list[StripPiece] | None = None
    merge_output: tuple[np.ndarray, np.ndarray] | None = None
    window_boxes: np.ndarray | None = None
    window_strip_shape: tuple[int, int] | None = None
    window_s: float | None = None
    window_output: tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None = None


def install_spies(capture_slot: list[_Capture]) -> None:
    """Wrap the two sealed functions; ``capture_slot[0]`` receives the records."""
    sealed_merge = augment_geometry.merge_strip_labels
    sealed_window = augment_geometry.place_window

    def merge_spy(pieces, center_index, *args, **kwargs):
        result = sealed_merge(pieces, center_index, *args, **kwargs)
        capture_slot[0].merge_pieces = list(pieces)
        capture_slot[0].merge_output = (result[0].copy(), result[1].copy())
        return result

    def window_spy(strip, boxes, s, *args, **kwargs):
        result = sealed_window(strip, boxes, s, *args, **kwargs)
        capture = capture_slot[0]
        capture.window_boxes = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        capture.window_strip_shape = strip.shape
        capture.window_s = s
        capture.window_output = result
        return result

    augment_geometry.merge_strip_labels = merge_spy
    augment_geometry.place_window = window_spy


# --------------------------------------------------- merge attribution


def _candidate(box: np.ndarray, class_id: int, piece: StripPiece) -> np.ndarray:
    """The box as it would appear in the merge output (band boxes column-cut)."""
    if class_id != BAND:
        return box
    cut = box.copy()
    cut[[0, 2]] = np.clip(cut[[0, 2]], piece.owned_x1_px, piece.owned_x2_px)
    return cut


def merge_attribution(
    pieces: Sequence[StripPiece], merged_boxes: np.ndarray, merged_classes: np.ndarray
) -> tuple[list[str], Counter]:
    """Provenance of each merged box and the reasons of the merge losses.

    Piece 0 is the centre crop (``assemble_context_strip`` puts it first).

    Returns:
        ``(provenance per merged box, Counter[(class name, reason)])``.
    """
    provenance: list[str | None] = [None] * len(merged_boxes)
    losses: Counter = Counter()
    center = pieces[0]
    neighbour_cells = augment_geometry._voronoi_cells(list(pieces[1:]))
    for index, piece in enumerate(pieces):
        role = CENTRE if index == 0 else NEIGHBOUR
        for box, class_value in zip(piece.boxes, piece.classes, strict=True):
            class_id = int(class_value)
            candidate = _candidate(box, class_id, piece)
            match = _first_unassigned_match(
                candidate, class_id, merged_boxes, merged_classes, provenance
            )
            if match is not None:
                provenance[match] = role
                continue
            reason = _merge_loss_reason(
                candidate, class_id, role, center, neighbour_cells, index
            )
            losses[(CLASS_NAMES[class_id], reason)] += 1
    # A box replaced by rule (가) appears under its new (neighbour) source.
    return [role or NEIGHBOUR for role in provenance], losses


def _first_unassigned_match(
    candidate: np.ndarray,
    class_id: int,
    merged_boxes: np.ndarray,
    merged_classes: np.ndarray,
    provenance: list[str | None],
) -> int | None:
    for position, (box, merged_class) in enumerate(
        zip(merged_boxes, merged_classes, strict=True)
    ):
        is_free = provenance[position] is None and int(merged_class) == class_id
        if is_free and np.array_equal(box, candidate):
            return position
    return None


def _merge_loss_reason(
    candidate: np.ndarray,
    class_id: int,
    role: str,
    center: StripPiece,
    neighbour_cells: list[tuple[float, float]],
    piece_index: int,
) -> str:
    if class_id == BAND:
        return "band_no_owned_columns" if candidate[2] <= candidate[0] else "band_other"
    if role == CENTRE:
        return "replaced_by_larger"  # rule (가); centre boxes are never dropped
    center_x = (candidate[0] + candidate[2]) / 2
    if center.x1_px <= center_x < center.x2_px:
        return "neighbour_in_centre_columns"
    low, high = neighbour_cells[piece_index - 1]
    if not low <= center_x < high:
        return "voronoi_other_neighbour"
    return "duplicate"


# -------------------------------------------------- window attribution


@dataclass
class ModeCounts:
    """Counts of one forced mode."""

    images: int = 0
    kept: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    merge_losses: Counter = field(default_factory=Counter)
    band_length_in: Counter = field(default_factory=Counter)
    band_length_below_half: Counter = field(default_factory=Counter)
    band_visible_share_below_half: Counter = field(default_factory=Counter)
    images_with_centre_band: int = 0
    images_with_band_drop: int = 0
    images_centre_band_all_lost: int = 0
    images_band_all_lost: int = 0
    images_with_labels: int = 0
    images_all_labels_lost: int = 0
    sources: Counter = field(default_factory=Counter)
    band_by_role: Counter = field(default_factory=Counter)
    """(centre | neighbour, kept | dropped) -> window-stage band label count."""


def _bin_label(value: float, edges: Sequence[float]) -> str:
    for low, high in itertools.pairwise(edges):
        if low <= value < high:
            return f"{low}-{high}"
    return f">={edges[-1]}"


def _scale_factors(strip_shape: tuple[int, int], s: float) -> tuple[float, float]:
    """The exact per-axis factors ``place_window`` uses (round of the new size)."""
    height_px, width_px = strip_shape
    if s == 1.0:
        return 1.0, 1.0
    return round(width_px * s) / width_px, round(height_px * s) / height_px


def window_reason(
    reference_area: float, clipped: np.ndarray, provenance: str, min_share: float
) -> tuple[str, float]:
    """Cause of one window-stage drop and its visible share of the scaled box."""
    visible_area = float(box_areas(clipped[None])[0])
    share = visible_area / reference_area if reference_area > 0 else 0.0
    if visible_area <= 0:
        return f"outside_window_{provenance}", share
    if share < min_share:
        return f"below_half_visible_{provenance}", share
    return "min_side", share


def record_image(
    counts: ModeCounts,
    capture: _Capture,
    input_classes: np.ndarray,
    record: Any,
    min_share: float,
) -> None:
    """Add one image's window-stage outcome to ``counts``."""
    counts.images += 1
    counts.sources[str(record.source)] += 1
    _, clipped, keep, _ = capture.window_output
    merged_classes = capture.merge_output[1] if capture.merge_pieces else None
    if capture.merge_pieces is not None:
        provenance, losses = merge_attribution(
            capture.merge_pieces, *capture.merge_output
        )
        counts.merge_losses.update(losses)
    else:
        provenance = [CENTRE] * len(capture.window_boxes)
        merged_classes = input_classes
    factor_x, factor_y = _scale_factors(capture.window_strip_shape, capture.window_s)
    reference_areas = box_areas(capture.window_boxes) * factor_x * factor_y
    any_band_dropped = False
    centre_band_kept = 0
    band_kept = 0
    for position, class_value in enumerate(np.asarray(merged_classes).reshape(-1)):
        name = CLASS_NAMES[int(class_value)]
        is_band = int(class_value) == BAND
        box = capture.window_boxes[position]
        if is_band:
            counts.band_length_in[
                _bin_label(box[2] - box[0], BAND_LENGTH_EDGES_PX)
            ] += 1
            outcome = "kept" if keep[position] else "dropped"
            counts.band_by_role[(provenance[position], outcome)] += 1
        if keep[position]:
            counts.kept[name] += 1
            band_kept += is_band
            centre_band_kept += is_band and provenance[position] == CENTRE
            continue
        counts.dropped[name] += 1
        reason, share = window_reason(
            reference_areas[position],
            clipped[position],
            provenance[position],
            min_share,
        )
        counts.reasons[(name, reason)] += 1
        if is_band:
            any_band_dropped = True
            if reason.startswith("below_half"):
                length = _bin_label(box[2] - box[0], BAND_LENGTH_EDGES_PX)
                counts.band_length_below_half[length] += 1
                counts.band_visible_share_below_half[
                    _bin_label(share, VISIBLE_SHARE_EDGES)
                ] += 1
    has_centre_band = bool((np.asarray(input_classes) == BAND).any())
    counts.images_with_band_drop += any_band_dropped
    if has_centre_band:
        counts.images_with_centre_band += 1
        counts.images_centre_band_all_lost += centre_band_kept == 0
        counts.images_band_all_lost += band_kept == 0
    if len(input_classes):
        counts.images_with_labels += 1
        counts.images_all_labels_lost += not bool(np.asarray(keep).any())


# ----------------------------------------------------------- summary


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _pairs_to_nested(counter: Counter) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for (name, reason), count in sorted(counter.items()):
        nested.setdefault(name, {})[reason] = count
    return nested


def mode_summary(counts: ModeCounts) -> dict:
    """JSON-friendly view of one mode."""
    per_class = {}
    for name in CLASS_NAMES:
        kept, dropped = counts.kept[name], counts.dropped[name]
        per_class[name] = {
            "kept": kept,
            "dropped": dropped,
            "dropped_share": _share(dropped, kept + dropped),
        }
    return {
        "images": counts.images,
        "context_source": dict(sorted(counts.sources.items())),
        "window_stage_by_class": per_class,
        "window_band_by_source": _pairs_to_nested(counts.band_by_role),
        "window_drop_reasons": _pairs_to_nested(counts.reasons),
        "merge_stage_losses": _pairs_to_nested(counts.merge_losses),
        "band_length_px_entering_window": dict(counts.band_length_in),
        "band_length_px_below_half_visible": dict(counts.band_length_below_half),
        "band_visible_share_below_half": dict(counts.band_visible_share_below_half),
        "images_with_band_drop": counts.images_with_band_drop,
        "images_with_band_drop_share": _share(
            counts.images_with_band_drop, counts.images
        ),
        "images_with_centre_band": counts.images_with_centre_band,
        "centre_band_all_lost_share": _share(
            counts.images_centre_band_all_lost, counts.images_with_centre_band
        ),
        "any_band_all_lost_share": _share(
            counts.images_band_all_lost, counts.images_with_centre_band
        ),
        "images_all_labels_lost_share": _share(
            counts.images_all_labels_lost, counts.images_with_labels
        ),
    }


def mixed_band_drop_share(context: ModeCounts, plain: ModeCounts) -> float | None:
    """Window-stage band drop share of an even mix (training p_context 0.5)."""
    kept = context.kept["pipe_band"] + plain.kept["pipe_band"]
    dropped = context.dropped["pipe_band"] + plain.dropped["pipe_band"]
    return _share(dropped, kept + dropped)


# ---------------------------------------------------------------- run


def _list_names(list_file: Path) -> list[str]:
    lines = list_file.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()).name for line in lines if line.strip()]


def _source_matches_seal() -> bool:
    completed = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", SEAL_COMMIT, "--", "src"],
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    return completed.returncode == 0


def run_mode(
    names: Sequence[str],
    p_context: float,
    seed: int,
    index: ContextIndex,
    settings: dict,
    capture_slot: list[_Capture],
) -> ModeCounts:
    """Push ``names`` through the sealed stage with the mode forced.

    ``capture_slot`` is the list the installed spies write into.
    """
    config = dataclasses.replace(
        context_config_from_dict(settings["gpr_context"]), p_context=p_context
    )
    keep_rule = geometric_config_from_dict(
        settings["gpr_geometric"], scale_owned_by_context=True
    ).keep_rule
    loader = FolderCropLoader(DATASET_DIR / "images", 256, 256)
    rng = np.random.default_rng(seed)
    counts = ModeCounts()
    for name in names:
        capture_slot[0] = _Capture()
        boxes, classes = loader.labels(name)
        content = CropContent(loader.image(name), boxes, classes)
        *_, record = augment_context_scale(
            name, content, index, loader, config, keep_rule, rng
        )
        record_image(
            counts,
            capture_slot[0],
            classes,
            record,
            keep_rule.min_visible_area_fraction,
        )
    return counts


def run(args: argparse.Namespace, identity: dict) -> int:
    if args.out.exists() and not args.overwrite:
        raise common.InputError(f"--out exists (use --overwrite): {args.out}")
    if not _source_matches_seal():
        raise common.InputError(f"src/ differs from the seal commit {SEAL_COMMIT}")
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    train = [
        name
        for k in common.FOLDS
        if k != args.fold
        for name in _list_names(DATASET_DIR / f"fold{k}_images.txt")
    ]
    validation = _list_names(DATASET_DIR / f"fold{args.fold}_images.txt")
    index_dir = REPO / settings["gpr_context"]["index_dir"]
    index = ContextIndex.from_csv(index_dir, train, validation)
    sample_rng = np.random.default_rng(args.seed)
    count = min(args.images, len(train))
    names = [train[i] for i in sorted(sample_rng.choice(len(train), count, False))]
    capture_slot = [_Capture()]
    install_spies(capture_slot)
    context = run_mode(names, 1.0, args.seed, index, settings, capture_slot)
    plain = run_mode(names, 0.0, args.seed, index, settings, capture_slot)
    primary = Counter(index.crop(name).primary_class for name in names)
    identity["finished_at"] = common.local_now()
    identity["seal_commit"] = SEAL_COMMIT
    identity["config_sha256"] = common.sha256_file(CONFIG)
    identity["index_sha256"] = index.file_sha256
    document = {
        "schema": SCHEMA,
        "note": "report only; augmentation label counts, not a detection metric",
        "run_identity": identity,
        "inputs": {
            "validation_fold": args.fold,
            "train_images": len(train),
            "sampled_images": count,
            "sampled_primary_class": dict(sorted(primary.items())),
            "seed": args.seed,
            "config": str(CONFIG),
        },
        "context_mode_forced": mode_summary(context),
        "plain_mode_forced": mode_summary(plain),
        "even_mix_window_band_drop_share": mixed_band_drop_share(context, plain),
    }
    common.write_json(args.out, document)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0, choices=common.FOLDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; 0 written, 2 input / seal problem, 3 internal error."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    args = _parse_args(argv)
    identity = common.new_run_identity()
    identity["gpu"] = "cpu"
    try:
        code = run(args, identity)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return common.print_final_line(2, args.out)
    except Exception:
        logger.exception("internal error")
        return common.print_final_line(3, args.out)
    return common.print_final_line(code, args.out)


if __name__ == "__main__":
    sys.exit(main())
