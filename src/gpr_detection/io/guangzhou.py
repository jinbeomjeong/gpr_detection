"""Guangzhou University ground-coupled GPR package (IDS .dt and MALA .rd3 as npy).

Facts used here (docs/datasets/readme_guangzhou.md, checked against the files):

* ``05_radargrams_npy/<target>/<mission_zone>/<file>.npy`` is int16
  ``[n_samples, n_traces]``; ``file_id`` is ``<target>/<mission_zone>/<file>``.
* ``dt_ns`` must be time window / sample count; the header ``data_y_cell`` is
  wrong for v3 files and the json ``dt_ns`` is rounded to 5 decimals.
* MALA files are already background-removed (and some gained), have no antenna
  frequency, and their ``t0_sample_used`` is a placeholder 0.
"""

from pathlib import Path

import numpy as np

from gpr_detection.io._common import optional_float, read_csv_as_text, read_json
from gpr_detection.io.radargram import Radargram, sample_to_ns

DATASET_KEY = "guangzhou"
NPY_FOLDER = "05_radargrams_npy"
INDEX_CSV = Path("01_메타데이터") / "radargrams_index.csv"
MALA_FORMAT = "MALA .rd3"
T0_USED_FIELD = "t0_sample_used"
MHZ_TO_HZ = 1e6


def list_guangzhou_files(root: Path, target: str | None = "pipe") -> list[str]:
    """Return file ids from radargrams_index.csv.

    Args:
        root: Guangzhou package root.
        target: ``"pipe"``, ``"rebar"``, or None for both.
    """
    index = read_csv_as_text(root / INDEX_CSV)
    if target is not None:
        index = index[index["target"] == target]
    return index["file_id"].tolist()


def is_mala_processed(meta: dict) -> bool:
    """True for MALA files, which are processed data rather than raw traces."""
    return meta["format"] == MALA_FORMAT


def load_guangzhou_radargram(root: Path, file_id: str) -> Radargram:
    """Load one Guangzhou profile as a :class:`Radargram`.

    Args:
        root: Guangzhou package root.
        file_id: e.g. ``"pipe/2020wate_200113AA/LID10002"``.

    Returns:
        Ground-coupled radargram. ``t0_candidates_ns`` holds ``t0_sample_used``
        (a K2 value that mixes direct-wave onset and peak; comparison only).
        MALA files get ``amplitude_processed=True`` and ``f_c_hz=None``.

    Raises:
        ValueError: If the array shape disagrees with the json.
    """
    stem = root / NPY_FOLDER / Path(file_id)
    meta = read_json(stem.with_suffix(".json"))
    raw = np.load(stem.with_suffix(".npy"))
    expected_shape = (meta["n_samples"], meta["n_traces"])
    if raw.shape != expected_shape:
        raise ValueError(f"{stem}.npy shape {raw.shape} != json {expected_shape}")

    dt_ns = float(meta["time_window_ns"]) / raw.shape[0]
    antenna_mhz = optional_float(meta["antenna_MHz"])
    is_mala = is_mala_processed(meta)
    return Radargram(
        data=raw.astype(np.float32),
        dt_ns=dt_ns,
        dx_m=float(meta["trace_dx_m"]),
        air_gap="ground",
        amplitude_processed=is_mala,
        dataset=DATASET_KEY,
        line_id=file_id,
        f_c_hz=None if antenna_mhz is None else antenna_mhz * MHZ_TO_HZ,
        eps_r_meta=None,
        t0_candidates_ns={T0_USED_FIELD: sample_to_ns(meta[T0_USED_FIELD], dt_ns)},
        metadata={**meta, "is_mala": is_mala},
    )
