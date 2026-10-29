"""Detection-image layer (L1) and its mapping to the radargram grid (L0).

``docs/architecture/README.md`` section 2-3 defines three coordinate layers:
L0 radargram grid (trace ``i``, sample ``k``), L1 detection image pixels
(``u``, ``v``) and L2 model input pixels. This module owns the L1 <-> L0 map.

Pixel convention used by every layer: coordinates are continuous and pixel
(or trace) ``j`` covers the half-open interval ``[j, j + 1)``. A box that
covers columns 10..19 therefore spans ``x1 = 10`` to ``x2 = 20``, which is
also how Ultralytics reports ``xyxy`` boxes.

The radargram-to-image conversion itself (``DetectionImage``) is added in M3-2;
until then AIHub and Morocco images use :meth:`ImageGridMap.identity` because
their pixels already are the grid.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageGridMap:
    """Linear map from detection-image pixels (L1) to the radargram grid (L0).

    ``trace = trace_offset + u * traces_per_px`` and
    ``sample = sample_offset + v * samples_per_px``. The two axes have separate
    factors because a resized B-scan need not keep a square pixel.

    Attributes:
        trace_offset: Grid trace at image column 0 (tile origin).
        sample_offset: Grid sample at image row 0 (tile origin).
        traces_per_px: Traces per image pixel along x.
        samples_per_px: Samples per image pixel along y.
        n_traces: Trace count of the whole radargram, for clipping.
        n_samples: Sample count of the whole radargram, for clipping.
    """

    trace_offset: int
    sample_offset: int
    traces_per_px: float
    samples_per_px: float
    n_traces: int
    n_samples: int

    def __post_init__(self) -> None:
        if self.traces_per_px <= 0 or self.samples_per_px <= 0:
            raise ValueError(f"grid factors must be positive, got {self}")
        if self.n_traces <= 0 or self.n_samples <= 0:
            raise ValueError(f"grid size must be positive, got {self}")
        if not (
            0 <= self.trace_offset < self.n_traces
            and 0 <= self.sample_offset < self.n_samples
        ):
            raise ValueError(f"tile origin must lie inside the grid, got {self}")

    @classmethod
    def identity(cls, width_px: int, height_px: int) -> "ImageGridMap":
        """Map for images whose pixels are the grid (AIHub, Morocco JPG).

        Args:
            width_px: Image width = number of traces.
            height_px: Image height = number of samples.
        """
        return cls(
            trace_offset=0,
            sample_offset=0,
            traces_per_px=1.0,
            samples_per_px=1.0,
            n_traces=width_px,
            n_samples=height_px,
        )
