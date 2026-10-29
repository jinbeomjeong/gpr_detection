"""4TU Netherlands air-launched GPR package (npy + json, metadata CSVs).

Facts used here (docs/datasets/readme_4tu.md, checked against the files):

* ``05_radargrams_npy/<activity>/<path>.npy`` is int16 ``[512, n_traces]``.
* ``dt_ns`` comes from the SEG-Y header in integer ps (0.097, 0.098, 0.117,
  0.195 ns); ``trace_dx_m`` is 0.02 m; the antenna is a 500 MHz air-launched unit.
* ``t0_sample_est`` / ``t0_sample_act_median`` are sample indices picked as the
  maximum mean amplitude; they may sit on the direct wave, not the surface
  reflection, so they are only candidates (spec_depth_v1 section 4-3).
"""

from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from gpr_detection.io._common import read_csv_as_text, read_json
from gpr_detection.io.radargram import Radargram, sample_to_ns

DATASET_KEY = "4tu"
CENTER_FREQUENCY_HZ = 500e6
EPS_R_SOURCE = "4tu_metadata_csv"
NPY_FOLDER = "05_radargrams_npy"
METADATA_FOLDER = "01_메타데이터"
T0_EST_FIELD = "t0_sample_est"
T0_ACT_MEDIAN_FIELD = "t0_sample_act_median"


class T0Choice(StrEnum):
    """Which metadata time-zero candidates to attach to the radargram."""

    EST = "est"
    ACT_MEDIAN = "act_median"
    BOTH = "both"


_T0_FIELDS_BY_CHOICE: dict[T0Choice, tuple[str, ...]] = {
    T0Choice.EST: (T0_EST_FIELD,),
    T0Choice.ACT_MEDIAN: (T0_ACT_MEDIAN_FIELD,),
    T0Choice.BOTH: (T0_EST_FIELD, T0_ACT_MEDIAN_FIELD),
}


class FourTUTable(StrEnum):
    """Metadata CSV files in ``01_메타데이터``."""

    GROUNDTRUTH = "utilities_groundtruth.csv"
    RADARGRAMS_INDEX = "radargrams_index.csv"
    ACTIVITIES = "activities_metadata_ko.csv"
    UTILITIES_METADATA = "utilities_metadata_list.csv"


def site_of_activity(activity_id: str) -> int:
    """Return the site number (1-13) of an activity id such as ``"010.14"``."""
    return int(activity_id.split(".", 1)[0])


def load_fourtu_table(root: Path, table: FourTUTable) -> pd.DataFrame:
    """Read one metadata CSV with all columns as text.

    Args:
        root: 4TU package root.
        table: Which CSV to read.
    """
    return read_csv_as_text(root / METADATA_FOLDER / table.value)


def list_fourtu_lines(root: Path) -> list[tuple[str, str]]:
    """Return ``(activity_id, path)`` for every radargram in radargrams_index.csv."""
    index = load_fourtu_table(root, FourTUTable.RADARGRAMS_INDEX)
    return list(zip(index["activity_id"], index["path"], strict=True))


def load_fourtu_radargram(
    root: Path,
    activity_id: str,
    path: str,
    t0_choice: T0Choice = T0Choice.BOTH,
) -> Radargram:
    """Load one 4TU profile as a :class:`Radargram`.

    Args:
        root: 4TU package root.
        activity_id: Activity folder, e.g. ``"01.1"``.
        path: Profile name, e.g. ``"Path3"`` or ``"Profile_6063"``.
        t0_choice: Metadata time-zero candidates to include in
            ``t0_candidates_ns`` (keys ``t0_sample_est``, ``t0_sample_act_median``).

    Returns:
        Air-coupled radargram with ``eps_r_meta`` from ``eps_r_used``.

    Raises:
        ValueError: If the array shape disagrees with the json.
    """
    stem = root / NPY_FOLDER / activity_id / path
    meta = read_json(stem.with_suffix(".json"))
    # Plain np.load: per-file memory maps were slower on Windows (decode_benchmark).
    raw = np.load(stem.with_suffix(".npy"))
    expected_shape = (meta["n_samples"], meta["n_traces"])
    if raw.shape != expected_shape:
        raise ValueError(f"{stem}.npy shape {raw.shape} != json {expected_shape}")

    dt_ns = float(meta["dt_ns"])
    t0_candidates = {
        field_name: sample_to_ns(meta[field_name], dt_ns)
        for field_name in _T0_FIELDS_BY_CHOICE[T0Choice(t0_choice)]
    }
    return Radargram(
        data=raw.astype(np.float32),
        dt_ns=dt_ns,
        dx_m=float(meta["trace_dx_m"]),
        air_gap="air",
        amplitude_processed=False,
        dataset=DATASET_KEY,
        line_id=f"{activity_id}/{path}",
        f_c_hz=CENTER_FREQUENCY_HZ,
        # eps_r_used replaces the broken header value (2.03) of 09.5 Path8/Path13.
        eps_r_meta=float(meta["eps_r_used"]),
        eps_r_meta_source=EPS_R_SOURCE,
        antenna_offset_m=None,
        t0_candidates_ns=t0_candidates,
        metadata={**meta, "site": site_of_activity(activity_id)},
    )
