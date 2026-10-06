"""Time allocation and rule checks for script outlines. Pure: no IO, no models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MIN_BEAT_S = 3
"""A beat shorter than this cannot carry a segment."""
OUTLINE_TOLERANCE = 0.2


def fit_durations(weights: Sequence[float], total_s: int, min_s: int = MIN_BEAT_S) -> list[int]:
    """Split `total_s` whole seconds in proportion to `weights`: the parts add up to exactly
    `total_s` and none is below `min_s`. A part that would fall below the minimum is held at it
    and the rest share what is left, still in proportion."""
    n = len(weights)
    if n == 0:
        raise ValueError("nothing to split")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    if total_s < min_s * n:
        raise ValueError(f"{total_s} s cannot give {n} parts at least {min_s} s each")

    held: set[int] = set()
    while True:
        free = [i for i in range(n) if i not in held]
        if not free:  # nothing but the minimums fits (total_s == min_s * n)
            scale = 0.0
            break
        scale = (total_s - min_s * len(held)) / sum(weights[i] for i in free)
        low = {i for i in free if weights[i] * scale < min_s}
        if not low:
            break
        held |= low
    exact = {i: weights[i] * scale for i in free}
    parts = [min_s] * n
    for i in free:
        parts[i] = int(exact[i])
    # hand out the seconds lost to flooring, biggest fractional part first (earlier on ties)
    order = sorted(free, key=lambda i: (-(exact[i] - parts[i]), i))
    for i in order[: total_s - sum(parts)]:
        parts[i] += 1
    return parts


@dataclass(frozen=True)
class BeatDraft:
    """A beat as a model proposed it, before it is known to be valid."""

    beat: str
    scene_refs: Sequence[str]
    target_s: float


def check_outline(
    beats: Sequence[BeatDraft],
    valid_scene_ids: Sequence[str],
    *,
    target_s: int,
    tolerance: float = OUTLINE_TOLERANCE,
) -> list[str]:
    """Rule violations of an outline draft, as messages to show the model."""
    valid = set(valid_scene_ids)
    errors: list[str] = []
    seen: set[str] = set()
    for n, b in enumerate(beats, 1):
        if b.beat in seen:
            errors.append(f"第 {n} 个节拍的名字 {b.beat!r} 重复")
        seen.add(b.beat)
        if not b.scene_refs:
            errors.append(f"第 {n} 个节拍没有 scene_refs")
        unknown = [r for r in b.scene_refs if r not in valid]
        if unknown:
            errors.append(f"第 {n} 个节拍引用了不存在的场景：{', '.join(unknown)}")
        if b.target_s < MIN_BEAT_S:
            errors.append(f"第 {n} 个节拍只有 {b.target_s} 秒，至少 {MIN_BEAT_S} 秒")
    total = sum(b.target_s for b in beats)
    lo, hi = round(target_s * (1 - tolerance)), round(target_s * (1 + tolerance))
    if not lo <= total <= hi:
        errors.append(f"各节拍时长合计 {total} 秒，目标 {target_s} 秒（允许 {lo}–{hi}）")
    return errors
