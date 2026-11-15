"""Deterministic stand-in detection backend for real-line PP-off checks.

Shared by ``tests/test_pp.py`` (§5 1a) and
``analysis/detection/m33_pp_off_identity.py`` (§5 1b, also run on the HEAD
a0dc987 tree before PP), so it imports only ``gpr_detection.detect.base``.
"""

import numpy as np

from gpr_detection.detect.base import Detection, DetectorOutput, PixelBox, letterbox

TILE_PX = 256
HYPERBOLA, BAND, CAVITY = 0, 1, 2


class TileStatisticsDetector:
    """Boxes placed from each tile's pixels (no model, same output every run).

    Per tile it reports a hyperbola at the brightest row, a band at the row of
    largest row energy and a cavity reaching into the padding, so a real line
    gets boxes near the zero band and the content end.
    """

    class_names = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")

    def predict(self, image: np.ndarray) -> DetectorOutput:
        _, transform = letterbox(image, TILE_PX)
        values = image.astype(np.float64)
        energy = ((values - values.mean()) ** 2).mean(axis=1)
        band_row = float(np.argmax(energy))
        bright_row = float(np.argmax(values.mean(axis=1)))
        score = float(values.mean() / 255.0)
        boxes = [
            (
                HYPERBOLA,
                score,
                PixelBox(40, bright_row, 90, min(bright_row + 30, TILE_PX)),
            ),
            (
                BAND,
                0.3 + 0.4 * score,
                PixelBox(0, band_row, 200, min(band_row + 8, TILE_PX)),
            ),
            (CAVITY, 0.2, PixelBox(120, 200, 160, TILE_PX)),
        ]
        detections = tuple(
            Detection(class_id, self.class_names[class_id], value, transform.apply(box))
            for class_id, value, box in boxes
        )
        return DetectorOutput(detections, transform)
