"""Scene segmentation planning (ARCHITECTURE §7.1, scenes). Pure: no IO, no models.

Scenes are found in two passes. This module is the first: *candidate* cuts, generously chosen
where the picture changes a lot between two shots and the dialogue pauses. The LLM then decides,
window by window, which candidates are real scene changes (the stage does that; the window
planning and the merging of its answers live here so they can be tested without a model)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from offscreen.algo.frames import histogram_similarity
from offscreen.domain.index import TranscriptLine

VISUAL_WEIGHT = 0.6
GAP_WEIGHT = 0.4
GAP_FULL_MS = 4_000
"""A pause this long (or longer) counts as the full dialogue signal."""
MIN_SCENE_MS = 20_000
"""Candidates closer than this to a stronger one are dropped; it is also the shortest scene."""
MAX_SPAN_MS = 6 * 60_000
"""A stretch without candidates longer than this gets its strongest boundary added."""
CANDIDATE_THRESHOLD = 0.45


@dataclass(frozen=True)
class Boundary:
    """The cut between shot `after` and shot `after + 1`."""

    after: int
    time_ms: int
    dissimilarity: float
    """1 - colour-histogram similarity of the last frame before and the first frame after."""
    gap_ms: int
    """Silence around the cut: 0 when somebody is speaking across it."""
    score: float


def dialogue_gap_ms(lines: Sequence[TranscriptLine], t_ms: int, duration_ms: int) -> int:
    """How long nobody speaks around time `t_ms`: from the end of the last line that ended by
    then to the start of the next line that starts at or after it (the film's ends bound it).
    0 if a line is being spoken across `t_ms`."""
    prev_end, next_start = 0, duration_ms
    for x in lines:
        if x.start_ms < t_ms < x.end_ms:
            return 0
        if x.end_ms <= t_ms:
            prev_end = max(prev_end, x.end_ms)
        elif x.start_ms >= t_ms:
            next_start = min(next_start, x.start_ms)
    return max(0, next_start - prev_end)


def boundary_features(
    shot_ends_ms: Sequence[int],
    first_hists: Sequence[Sequence[int]],
    last_hists: Sequence[Sequence[int]],
    lines: Sequence[TranscriptLine],
    duration_ms: int,
) -> list[Boundary]:
    """A `Boundary` for every pair of neighbouring shots. `first_hists[i]` / `last_hists[i]` are
    the colour histograms of shot `i`'s first / last keyframe."""
    n = len(shot_ends_ms)
    if not (len(first_hists) == len(last_hists) == n):
        raise ValueError("per-shot inputs must have the same length")
    out: list[Boundary] = []
    for i in range(n - 1):
        dissim = 1.0 - histogram_similarity(last_hists[i], first_hists[i + 1])
        gap = dialogue_gap_ms(lines, shot_ends_ms[i], duration_ms)
        score = VISUAL_WEIGHT * dissim + GAP_WEIGHT * min(gap, GAP_FULL_MS) / GAP_FULL_MS
        out.append(Boundary(i, shot_ends_ms[i], dissim, gap, score))
    return out


def pick_candidates(
    boundaries: Sequence[Boundary],
    *,
    threshold: float = CANDIDATE_THRESHOLD,
    min_gap_ms: int = MIN_SCENE_MS,
    max_span_ms: int = MAX_SPAN_MS,
    duration_ms: int,
) -> list[int]:
    """Indices (`Boundary.after`) of the cuts worth asking the model about, ascending.

    Boundaries scoring at least `threshold` are taken strongest first, skipping any within
    `min_gap_ms` of one already taken (so no candidate scene is shorter than that). Then any
    stretch longer than `max_span_ms` gets its strongest eligible boundary, until none is left."""
    chosen: list[Boundary] = []

    def room_for(b: Boundary) -> bool:
        return all(abs(b.time_ms - c.time_ms) >= min_gap_ms for c in chosen)

    for b in sorted(boundaries, key=lambda b: (-b.score, b.after)):
        if b.score >= threshold and room_for(b):
            chosen.append(b)

    while True:
        edges = [0, *sorted(c.time_ms for c in chosen), duration_ms]
        wide = [(a, z) for a, z in pairwise(edges) if z - a > max_span_ms]
        if not wide:
            break
        added = False
        for a, z in wide:
            inside = [
                b for b in boundaries if a < b.time_ms < z and room_for(b) and b not in chosen
            ]
            if inside:
                chosen.append(max(inside, key=lambda b: (b.score, -b.after)))
                added = True
                break
        if not added:
            break  # nothing eligible: the long stretch stays as it is
    return sorted(b.after for b in chosen)


def segments_from_cuts(n_shots: int, cuts: Sequence[int]) -> list[tuple[int, int]]:
    """Shot index ranges `[lo, hi)` between candidate cuts (`cuts` = `Boundary.after` values),
    partitioning `0 .. n_shots`."""
    if n_shots <= 0:
        return []
    starts = [0, *(c + 1 for c in sorted(set(cuts)) if 0 <= c < n_shots - 1)]
    return [(lo, hi) for lo, hi in zip(starts, [*starts[1:], n_shots], strict=True)]


@dataclass(frozen=True)
class Window:
    """Segments `[lo, hi)` are shown to the model together; of the joins inside them (join `j`
    sits between segment `j` and `j + 1`) it is trusted for `[accept_lo, accept_hi)`."""

    lo: int
    hi: int
    accept_lo: int
    accept_hi: int


def plan_windows(n_segments: int, size: int, overlap: int) -> list[Window]:
    """Overlapping windows over the segments. Every join is decided by exactly one window, and
    always inside that window's own joins: neighbouring windows hand over in the middle of their
    overlap, so a join is decided by a window that sees segments on both sides of it."""
    if size < 2 or not 1 <= overlap < size:
        raise ValueError("need size >= 2 and 1 <= overlap < size")
    joins = n_segments - 1
    if joins <= 0:
        return []
    if n_segments <= size:
        return [Window(0, n_segments, 0, joins)]
    stride = size - overlap
    los = list(range(0, n_segments - size + 1, stride))
    if los[-1] + size < n_segments:  # the last window must reach the end
        los.append(n_segments - size)
    windows: list[Window] = []
    accept_lo = 0
    for k, lo in enumerate(los):
        hi = lo + size
        if k + 1 < len(los):
            shared = hi - los[k + 1]  # segments both windows show (>= overlap)
            accept_hi = max(accept_lo, los[k + 1] + shared // 2)
        else:
            accept_hi = joins
        windows.append(Window(lo, hi, accept_lo, accept_hi))
        accept_lo = accept_hi
    return windows


def merge_segments(
    segments: Sequence[tuple[int, int]], new_scene_after: Sequence[bool]
) -> list[tuple[int, int]]:
    """Scenes (shot ranges) from segments and the decision at every join: `new_scene_after[j]` is
    True when a scene starts at segment `j + 1`."""
    if len(new_scene_after) != max(0, len(segments) - 1):
        raise ValueError("one decision per join")
    if not segments:
        return []
    scenes: list[tuple[int, int]] = []
    lo = segments[0][0]
    for (_, hi), brk in zip(segments, [*new_scene_after, True], strict=True):
        if brk:
            scenes.append((lo, hi))
            lo = hi
    return scenes


def enforce_min_length(
    spans: Sequence[tuple[int, int]], starts_ms: Sequence[int], ends_ms: Sequence[int], min_ms: int
) -> list[tuple[int, int]]:
    """Fold scenes shorter than `min_ms` into a neighbour (the previous one; the first scene
    into the next), so the model's decisions can never leave a sliver. `starts_ms[i]` /
    `ends_ms[i]` are shot `i`'s times."""
    out = [list(s) for s in spans]
    i = 0
    while i < len(out) and len(out) > 1:
        lo, hi = out[i]
        if ends_ms[hi - 1] - starts_ms[lo] >= min_ms:
            i += 1
            continue
        if i > 0:
            out[i - 1][1] = hi
            del out[i]
            i = max(0, i - 1)
        else:
            out[1][0] = lo
            del out[0]
    return [(lo, hi) for lo, hi in out]
