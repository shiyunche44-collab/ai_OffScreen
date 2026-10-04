"""Naive shot picking for the v0 plan: order candidate shots, then fill a duration. Pure.

Real scoring and speed-based fitting (ARCHITECTURE §7.3 steps ②–④) replace this in M5; the
contract stays: clips are cut from the given shots, and their lengths add up to the target."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

MAX_CLIP_MS = 4_000
MIN_CLIP_MS = 800


@dataclass(frozen=True)
class Span:
    """A candidate shot: `[start_ms, end_ms)` of the source."""

    shot_id: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms >= self.end_ms:
            raise ValueError(f"empty span {self.shot_id}: {self.start_ms}..{self.end_ms}")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class ClipSpan:
    shot_id: str
    src_in_ms: int
    src_out_ms: int

    @property
    def duration_ms(self) -> int:
        return self.src_out_ms - self.src_in_ms


def order_candidates(shots: Sequence[Span], used: Collection[str]) -> list[Span]:
    """Chronological, with shots not used by earlier segments ahead of the ones that were."""
    ordered = sorted(shots, key=lambda s: (s.start_ms, s.end_ms))
    return [s for s in ordered if s.shot_id not in used] + [s for s in ordered if s.shot_id in used]


def _middle(span: Span, length: int) -> ClipSpan:
    """`length` ms from the middle of the shot (its start and end are the least telling)."""
    start = span.start_ms + (span.duration_ms - length) // 2
    return ClipSpan(span.shot_id, start, start + length)


def fill_clips(
    candidates: Sequence[Span],
    target_ms: int,
    *,
    max_clip_ms: int = MAX_CLIP_MS,
    min_clip_ms: int = MIN_CLIP_MS,
) -> list[ClipSpan]:
    """Clips taken from `candidates` in order (wrapping around when they run out) whose
    lengths add up to exactly `target_ms`.

    Each clip is at most `max_clip_ms` and comes from the middle of its shot. The last clip is
    never left shorter than `min_clip_ms` when the shot before it can give up some time; a
    clip is only shorter than that when its shot is, or when `target_ms` itself is."""
    if target_ms <= 0:
        return []
    if not candidates:
        raise ValueError("no candidate shots")
    clips: list[ClipSpan] = []
    acc = 0
    i = 0
    while acc < target_ms:
        span = candidates[i % len(candidates)]
        i += 1
        take = min(span.duration_ms, max_clip_ms)
        rem = target_ms - acc
        if take >= rem:
            take = rem
        elif rem - take < min_clip_ms and rem - min_clip_ms >= min_clip_ms:
            take = rem - min_clip_ms  # leave a tail the next clip can fill decently
        clips.append(_middle(span, take))
        acc += take
    return clips
