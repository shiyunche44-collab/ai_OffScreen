"""From a detector's per-frame transition scores to shots. Pure: scores in, frames out."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise


def cut_frames(scores: Sequence[float], threshold: float = 0.5) -> list[int]:
    """The first frame of each new shot, from per-frame scores where a high score at frame `i`
    says "the picture changes after frame `i`" (TransNetV2's convention). A run of high scores
    is one transition (a dissolve spans several frames); the new shot starts right after the
    run. A run reaching the last frame cuts nothing: no shot follows it."""
    cuts: list[int] = []
    run_end: int | None = None
    for i, s in enumerate(scores):
        if s > threshold:
            run_end = i
            continue
        if run_end is not None:
            cuts.append(run_end + 1)
            run_end = None
    return cuts


def spans_from_cuts(
    cuts: Sequence[int], frame_count: int, fps_num: int, fps_den: int
) -> list[tuple[int, int]]:
    """`(start_ms, end_ms)` of the shots a list of cut frames makes in a film of `frame_count`
    frames: the first shot starts at 0, the last ends with the film. Cuts outside the film and
    repeats are ignored."""
    edges = [0, *sorted({c for c in cuts if 0 < c < frame_count}), frame_count]

    def ms(frame: int) -> int:
        return (frame * fps_den * 1000 + fps_num // 2) // fps_num

    return [(ms(a), ms(b)) for a, b in pairwise(edges) if ms(b) > ms(a)]
