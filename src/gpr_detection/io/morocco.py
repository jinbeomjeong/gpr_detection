"""Morocco pipe/cavity YOLO package (screen-captured patches, no physical scale).

Facts used here (docs/datasets/readme_morocco.md, checked against the files):

* ``03_yolo_dataset/{images,labels}/{train,val,test}`` with ``data.yaml``
  names ``0: pipe, 1: cavity``; intact images have empty label files.
* ``01_메타데이터/images_index.csv`` lists 2,239 augmented images (in the YOLO
  folder) and 285 originals (``05_원본패치``, no labels). ``group_id`` is the
  provided leak-free split group; ``aug_transform`` names the augmentation.
* Images are 224, 268 or 336 px square 8-bit grayscale JPGs.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from gpr_detection.io._common import read_csv_as_text, read_grayscale_file
from gpr_detection.io.annotated import AnnotatedImage, Box
from gpr_detection.io.classes import MOROCCO_EXPECTED_NAMES, map_morocco_class_id

DATASET_KEY = "morocco"
YOLO_FOLDER = "03_yolo_dataset"
INDEX_CSV = Path("01_메타데이터") / "images_index.csv"
SPLIT_GROUPS_CSV = Path("01_메타데이터") / "split_groups.csv"
GROUP_STEMS_SEPARATOR = " | "
SPLITS = ("train", "val", "test")
# Rotation (회전 ±15°) and elastic warp (탄성 변형) distort hyperbola geometry and
# are excluded by the augmentation policy (docs/experiments/augmentation_policy.md).
GEOMETRY_DISTORTING_TRANSFORM_PREFIXES = ("회전", "탄성")
YOLO_FIELDS_PER_LINE = 5


@dataclass(frozen=True)
class MoroccoImageRef:
    """One row of images_index.csv for an image inside the YOLO folder.

    Attributes:
        image_id: e.g. ``"U_001_aug_1"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        subset: ``"utilities"``, ``"cavities"`` or ``"intact"``.
        stem: Original patch stem shared by all its augmentations.
        group_id: Provided split group (near-duplicates share one group).
        aug_transform: Augmentation description in Korean.
        image_relpath: Image path relative to the package root (POSIX style).
    """

    image_id: str
    split: str
    subset: str
    stem: str
    group_id: str
    aug_transform: str
    image_relpath: str

    @property
    def label_relpath(self) -> str:
        """YOLO label path relative to the package root."""
        image_path = PurePosixPath(self.image_relpath)
        split_folder = image_path.parent.name
        return str(
            image_path.parents[2] / "labels" / split_folder / f"{image_path.stem}.txt"
        )


def read_morocco_class_names(root: Path) -> dict[int, str]:
    """Return ``names`` from data.yaml."""
    config = yaml.safe_load(
        (root / YOLO_FOLDER / "data.yaml").read_text(encoding="utf-8")
    )
    return {int(class_id): name for class_id, name in config["names"].items()}


def _check_class_names(root: Path) -> None:
    names = read_morocco_class_names(root)
    if names != MOROCCO_EXPECTED_NAMES:
        raise ValueError(
            f"Morocco data.yaml names {names} != expected {MOROCCO_EXPECTED_NAMES}; "
            "update gpr_detection.io.classes before loading"
        )


def is_geometry_distorting(aug_transform: str) -> bool:
    """True for rotation and elastic augmentations."""
    return aug_transform.startswith(GEOMETRY_DISTORTING_TRANSFORM_PREFIXES)


def list_morocco_images(
    root: Path,
    split: str | None = None,
    drop_rotated_and_elastic: bool = False,
) -> list[MoroccoImageRef]:
    """List labelled images of the YOLO folder.

    Args:
        root: Morocco package root.
        split: Keep only this split; None keeps all three.
        drop_rotated_and_elastic: Exclude rotation and elastic augmentations.

    Raises:
        ValueError: On an unknown split or unexpected data.yaml class names.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
    _check_class_names(root)

    index = read_csv_as_text(root / INDEX_CSV)
    # Originals live in 05_원본패치 and have no labels; keep only YOLO images.
    index = index[index["yolo_path"].str.startswith(f"{YOLO_FOLDER}/")]
    if split is not None:
        index = index[index["split"] == split]

    refs = [
        MoroccoImageRef(
            image_id=row.image_id,
            split=row.split,
            subset=row.subset,
            stem=row.stem,
            group_id=row.group_id,
            aug_transform=row.aug_transform,
            image_relpath=row.yolo_path,
        )
        for row in index.itertuples(index=False)
    ]
    if drop_rotated_and_elastic:
        refs = [ref for ref in refs if not is_geometry_distorting(ref.aug_transform)]
    return refs


def read_morocco_group_stems(root: Path) -> dict[str, tuple[str, ...]]:
    """Return the original-patch stems of each split group, in file order.

    ``split_groups.csv`` lists the stems as ``"177 | 179 | ..."``; the first
    stem is the one the ``group_id`` is named after.
    """
    groups = read_csv_as_text(root / SPLIT_GROUPS_CSV)
    return {
        row.group_id: tuple(row.stems.split(GROUP_STEMS_SEPARATOR))
        for row in groups.itertuples(index=False)
    }


def _parse_yolo_line(line: str, width: int, height: int, label_path: Path) -> Box:
    fields = line.split()
    if len(fields) != YOLO_FIELDS_PER_LINE:
        raise ValueError(f"{label_path}: expected 5 fields, got {line!r}")
    native_id = int(fields[0])
    x_center, y_center, box_width, box_height = (float(value) for value in fields[1:])
    return Box(
        class_id=map_morocco_class_id(native_id),
        x1=(x_center - box_width / 2) * width,
        y1=(y_center - box_height / 2) * height,
        x2=(x_center + box_width / 2) * width,
        y2=(y_center + box_height / 2) * height,
        source_label=MOROCCO_EXPECTED_NAMES[native_id],
    )


def load_morocco_image(root: Path, ref: MoroccoImageRef) -> AnnotatedImage:
    """Decode one image and convert its normalised YOLO boxes to pixels.

    Returns:
        Annotated image with ``session=group_id`` and
        ``crop_key="<subset>/<stem>"`` (all augmentations of one original).

    Raises:
        FileNotFoundError: If the label file is missing.
    """
    image = read_grayscale_file(root / ref.image_relpath)
    height, width = image.shape
    label_path = root / ref.label_relpath
    label_lines = label_path.read_text(encoding="utf-8").splitlines()
    boxes = tuple(
        _parse_yolo_line(line, width, height, label_path)
        for line in label_lines
        if line.strip()
    )
    return AnnotatedImage(
        image=image,
        boxes=boxes,
        dataset=DATASET_KEY,
        image_id=ref.image_id,
        session=ref.group_id,
        crop_key=f"{ref.subset}/{ref.stem}",
        metadata={
            "split": ref.split,
            "subset": ref.subset,
            "aug_transform": ref.aug_transform,
        },
    )
