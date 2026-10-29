"""GPR-SD (USTC) road GPR package: processed float32 B-scans with two dt hypotheses.

Facts used here (docs/datasets/readme_gprsd.md, checked against the files):

* ``05_radargrams_npy/<road>_<line>.npy`` is float32 ``[321, n_traces]``,
  ``trace_dx_m`` 0.03 m. Surface echo removal and gain are already applied.
* The time axis is unconfirmed: hypothesis A treats the 321 samples as the
  documented 97 ns window (0.3022 ns), hypothesis B assumes 59 leading samples
  of a 380-sample record were cut (0.2553 ns). Curvature depth does not depend
  on the choice; permittivity does (spec_depth_v1 section 7).
* Amplitudes reach about 1.4e5, so a symmetric |amplitude| percentile clip is
  offered for the detection branch.
"""

from enum import StrEnum
from pathlib import Path

import numpy as np

from gpr_detection.io._common import read_csv_as_text, read_json
from gpr_detection.io.radargram import Radargram

DATASET_KEY = "gprsd"
NPY_FOLDER = "05_radargrams_npy"
INDEX_CSV = Path("01_메타데이터") / "radargrams_index.csv"


class DtHypothesis(StrEnum):
    """Sample-interval hypothesis for GPR-SD (see module docstring)."""

    A = "A"
    B = "B"


_DT_FIELD_BY_HYPOTHESIS: dict[DtHypothesis, str] = {
    DtHypothesis.A: "dt_ns_hypA_321",
    DtHypothesis.B: "dt_ns_hypB_380",
}


def list_gprsd_files(root: Path) -> list[str]:
    """Return the 12 file ids (``<road>_<line>``) from radargrams_index.csv."""
    return read_csv_as_text(root / INDEX_CSV)["file_id"].tolist()


def clip_abs_percentile(
    data: np.ndarray, percentile: float
) -> tuple[np.ndarray, float]:
    """Clip ``data`` symmetrically to the given percentile of ``|data|``.

    Returns:
        The clipped array (float32) and the clip level.
    """
    clip_level = float(np.percentile(np.abs(data), percentile))
    return np.clip(data, -clip_level, clip_level).astype(np.float32), clip_level


def load_gprsd_radargram(
    root: Path,
    file_id: str,
    dt_hypothesis: DtHypothesis,
    clip_percentile: float | None = None,
) -> Radargram:
    """Load one GPR-SD profile as a :class:`Radargram`.

    Args:
        root: GPR-SD package root.
        file_id: e.g. ``"CementRoad1_1"``.
        dt_hypothesis: Which sample interval to use. There is deliberately no
            default: depth work must run both and compare.
        clip_percentile: If given (e.g. 99), clip to that percentile of
            ``|amplitude|``. Meant for the detection branch only.

    Returns:
        Ground-coupled, amplitude-processed radargram. The applied hypothesis and
        clip level are recorded in ``metadata``.

    Raises:
        ValueError: If the array shape disagrees with the json.
    """
    stem = root / NPY_FOLDER / file_id
    meta = read_json(stem.with_suffix(".json"))
    data = np.load(stem.with_suffix(".npy"))
    expected_shape = (meta["n_samples"], meta["n_traces"])
    if data.shape != expected_shape:
        raise ValueError(f"{stem}.npy shape {data.shape} != json {expected_shape}")

    data = data.astype(np.float32, copy=False)
    clip_level = None
    if clip_percentile is not None:
        data, clip_level = clip_abs_percentile(data, clip_percentile)

    hypothesis = DtHypothesis(dt_hypothesis)
    return Radargram(
        data=data,
        dt_ns=float(meta[_DT_FIELD_BY_HYPOTHESIS[hypothesis]]),
        dx_m=float(meta["trace_dx_m"]),
        air_gap="ground",
        amplitude_processed=True,
        dataset=DATASET_KEY,
        line_id=file_id,
        f_c_hz=None,
        eps_r_meta=None,
        metadata={
            **meta,
            "dt_hypothesis": hypothesis.value,
            "clip_percentile": clip_percentile,
            "clip_level": clip_level,
        },
    )
