"""Helpers for searching shots. Pure: no IO, no models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from offscreen.domain.index import ShotCaption

RRF_K = 60
"""Reciprocal rank fusion constant: how fast the value of a rank falls off."""


def search_text(caption: ShotCaption) -> str:
    """The sentence a shot is found by: what is seen, plus the action and the mood."""
    parts = [caption.caption.strip()]
    if caption.action and caption.action.strip():
        parts.append(f"动作：{caption.action.strip()}")
    if caption.emotion and caption.emotion.strip():
        parts.append(f"情绪：{caption.emotion.strip()}")
    return "。".join(p.rstrip("。") for p in parts if p)


def fuse(rankings: Mapping[str, Sequence[str]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge several best-first lists of shot ids into one by reciprocal rank fusion: a shot
    scores the sum of `1 / (k + rank)` over the lists it appears in (rank from 1). Best first;
    ties go to the shot that appears earlier in the first list that has it."""
    score: dict[str, float] = {}
    first: dict[str, tuple[int, int]] = {}
    for n, ids in enumerate(rankings.values()):
        for rank, shot_id in enumerate(ids, start=1):
            score[shot_id] = score.get(shot_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(shot_id, (n, rank))
    return sorted(score.items(), key=lambda kv: (-kv[1], first[kv[0]]))
