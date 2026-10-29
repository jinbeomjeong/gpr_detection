"""Small readers shared by the dataset loaders."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read a metadata CSV with every column as str (empty cells stay ``""``).

    The dataset packages write CSVs as UTF-8 with BOM; text dtype keeps
    identifiers such as ``"01.1"`` and ``"001"`` from being parsed as numbers.
    """
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def decode_grayscale(encoded: bytes | np.ndarray, description: str) -> np.ndarray:
    """Decode JPG/PNG bytes to a 2-D uint8 image with cv2 IMREAD_GRAYSCALE.

    cv2 IMREAD_GRAYSCALE was the fastest bit-exact CPU decoder in
    docs/experiments/decode_benchmark.md. Decoding from memory also avoids
    cv2.imread failing on the non-ASCII dataset paths on Windows.

    Raises:
        ValueError: If OpenCV cannot decode the data.
    """
    buffer = (
        np.frombuffer(encoded, dtype=np.uint8)
        if isinstance(encoded, bytes)
        else encoded
    )
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {description}")
    return image


def read_grayscale_file(path: Path) -> np.ndarray:
    """Read an image file (non-ASCII path safe) as a 2-D uint8 array."""
    return decode_grayscale(np.fromfile(path, dtype=np.uint8), str(path))


def optional_float(value: Any) -> float | None:
    """Return ``float(value)``, or None for ``None`` / empty string."""
    if value is None or value == "":
        return None
    return float(value)
