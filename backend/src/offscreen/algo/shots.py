"""Shot list post-processing (ARCHITECTURE §7.1): merge fragments, split long shots."""

from __future__ import annotations

from itertools import pairwise

MIN_SHOT_MS = 500
MAX_SHOT_MS = 8000

Span = tuple[int, int]
"""Half-open source range `(start_ms, end_ms)`."""


def normalize_shots(
    raw: list[Span],
    duration_ms: int,
    *,
    min_ms: int = MIN_SHOT_MS,
    max_ms: int = MAX_SHOT_MS,
) -> list[Span]:
    """Turn raw detector output into a partition of `[0, duration_ms)`.

    1. Only the cut points are kept (each raw start after the first), clamped into the
       video, so gaps, overlaps and a short or long final shot in `raw` cannot leak out.
    2. Shots shorter than `min_ms` are merged into a neighbour.
    3. Shots longer than `max_ms` are split into equal-ish parts, which is easier to pick
       from when editing. Parts of a split are at least `max_ms / 2` long.

    Result: contiguous, starts at 0, ends at `duration_ms`, every shot <= `max_ms`, and
    every shot >= `min_ms` unless the whole video is shorter than that.
    """
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if not 0 < min_ms <= max_ms // 2:
        raise ValueError("need 0 < min_ms <= max_ms / 2")

    cuts = sorted({s for s, _ in raw[1:] if 0 < s < duration_ms})
    bounds = [0, *cuts, duration_ms]
    shots = list(pairwise(bounds))
    return _split_long(_merge_short(shots, min_ms), max_ms)


def _merge_short(shots: list[Span], min_ms: int) -> list[Span]:
    """Single pass. Every kept shot except possibly the first is >= min_ms, and the first
    keeps absorbing its successor until it is too, so only a video shorter than `min_ms`
    can end up with a short shot."""
    out: list[Span] = []
    for start, end in shots:
        if out and (end - start < min_ms or out[-1][1] - out[-1][0] < min_ms):
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def _split_long(shots: list[Span], max_ms: int) -> list[Span]:
    out: list[Span] = []
    for start, end in shots:
        n = -(-(end - start) // max_ms)  # ceil
        edges = [start + (i * (end - start)) // n for i in range(n)]
        out.extend(zip(edges, [*edges[1:], end], strict=True))
    return out
