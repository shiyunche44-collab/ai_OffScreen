"""Keyframe sampling and image-quality metrics (ARCHITECTURE §7.1, keyframes).

Pure functions: where to sample a shot, and how sharp / bright a decoded grayscale frame is, so
later stages can skip blurry or black frames when picking footage."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from offscreen.domain.index import ShotQuality

KEYFRAME_POSITIONS = (0.1, 0.5, 0.9)
"""Where in a shot the frames are taken, as a fraction of its duration."""

SHARPNESS_SCALE = 100.0
"""Laplacian variance at which the sharpness score is 0.5 (see `sharpness_score`)."""

Gray = NDArray[np.uint8]


def frame_times(
    start_ms: int, end_ms: int, positions: Sequence[float] = KEYFRAME_POSITIONS
) -> list[int]:
    """Source times to sample inside the half-open range `[start_ms, end_ms)`, one per position
    (rounded to whole ms). Always inside the range, and never decreasing."""
    if end_ms <= start_ms:
        raise ValueError("empty range")
    last = end_ms - 1
    times = [min(last, start_ms + round((end_ms - start_ms) * p)) for p in positions]
    return [max(start_ms, t) for t in times]


def laplacian_variance(gray: Gray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior pixels: large for fine detail,
    near zero for flat or blurred frames. Zero for images smaller than 3x3."""
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    g = gray.astype(np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def sharpness_score(variance: float) -> float:
    """Squash the unbounded variance into 0..1: `v / (v + SHARPNESS_SCALE)`, monotone, 0 for a
    flat frame and approaching 1 for very detailed ones."""
    if variance <= 0:
        return 0.0
    return variance / (variance + SHARPNESS_SCALE)


def brightness_score(gray: Gray) -> float:
    """Mean luma as 0..1 (0 = black frame, 1 = white)."""
    if gray.size == 0:
        return 0.0
    return float(gray.mean()) / 255.0


def frame_quality(gray: Gray) -> tuple[float, float]:
    """`(sharpness, brightness)` of one decoded grayscale frame."""
    return sharpness_score(laplacian_variance(gray)), brightness_score(gray)


HIST_BINS = 4
"""Colour histogram bins per channel: 4 x 4 x 4 = 64 bins, coarse enough that lighting changes
inside a scene barely move it, fine enough that a change of place usually does."""


def color_histogram(rgb: NDArray[np.uint8], bins: int = HIST_BINS) -> list[int]:
    """Share of pixels in each colour bin, in per-mille (the bins sum to 1000 up to rounding).
    `rgb` has shape `(height, width, 3)`; bin index is `r * bins * bins + g * bins + b`."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("expected a non-empty (height, width, 3) array")
    q = (rgb.astype(np.int64) * bins) // 256  # 0 .. bins-1 per channel
    index = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    counts = np.bincount(index.ravel(), minlength=bins**3)
    per_mille = counts * 1000 / counts.sum()
    out = np.floor(per_mille).astype(np.int64)
    out[int(np.argmax(counts))] += 1000 - int(
        out.sum()
    )  # give the rounding loss to the biggest bin
    return [int(v) for v in out]


def histogram_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Histogram intersection of two per-mille histograms: 1.0 for the same colour distribution,
    0.0 for completely different ones."""
    if len(a) != len(b):
        raise ValueError("histograms of different sizes")
    total = max(sum(a), sum(b))
    if total <= 0:
        return 0.0
    return sum(min(x, y) for x, y in zip(a, b, strict=True)) / total


def shot_quality(frames: Sequence[tuple[float, float]]) -> ShotQuality:
    """A shot's quality is the mean of its frames' `(sharpness, brightness)`."""
    if not frames:
        raise ValueError("a shot needs at least one frame")
    n = len(frames)
    return ShotQuality(
        sharpness=min(1.0, sum(f[0] for f in frames) / n),
        brightness=min(1.0, sum(f[1] for f in frames) / n),
    )


def sprite_slot(index: int, columns: int, rows: int) -> tuple[int, int, int]:
    """`(sheet, col, row)` of the `index`-th tile when sheets of `columns x rows` are filled
    left to right, top to bottom."""
    if index < 0 or columns <= 0 or rows <= 0:
        raise ValueError("bad sprite layout")
    per_sheet = columns * rows
    sheet, within = divmod(index, per_sheet)
    return sheet, within % columns, within // columns
