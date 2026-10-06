"""Pure helpers for the faces stage."""

from __future__ import annotations

import math
from collections.abc import Sequence


def box_area(bbox: Sequence[float]) -> float:
    """Share of the image covered by `(x0, y0, x1, y1)` (fractions of the image)."""
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def unit_vector(v: Sequence[float]) -> list[float] | None:
    """`v` scaled to length 1, or None for a zero / non-finite vector (nothing to compare).
    Scaled by the largest component first, so tiny or huge values neither underflow nor overflow
    while squaring."""
    if not v:
        return None
    largest = max(abs(x) for x in v)
    if not math.isfinite(largest) or largest == 0.0:
        return None
    scaled = [x / largest for x in v]
    norm = math.sqrt(sum(x * x for x in scaled))
    return [x / norm for x in scaled]
