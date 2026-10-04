"""Coarse scene planning for the v0 story stage. Pure: no IO, no models.

Dialogue is grouped into chunks that fit one LLM request; each chunk becomes one scene.
Scene borders are snapped to shot starts, so the scenes partition the film's shots and a
scene never splits a shot."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.domain.index import TranscriptLine

SOFT_GAP_MS = 4_000
"""A pause at least this long is a preferred place to break once a chunk is half full."""


@dataclass(frozen=True)
class ScenePlan:
    """Lines `lines[line_lo:line_hi]` and shots `shots[shot_lo:shot_hi]` form one scene."""

    line_lo: int
    line_hi: int
    shot_lo: int
    shot_hi: int


def fmt_clock(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def chunk_lines(
    lines: Sequence[TranscriptLine], *, max_chars: int, max_span_ms: int
) -> list[tuple[int, int]]:
    """Consecutive `[lo, hi)` index ranges over `lines` (assumed time-ordered) that cover
    every line exactly once. A chunk closes when the next line would push it past
    `max_chars` characters or `max_span_ms` of time; past half of either limit it already
    closes at a long pause."""
    if max_chars < 1 or max_span_ms < 1:
        raise ValueError("limits must be positive")
    chunks: list[tuple[int, int]] = []
    lo = 0
    chars = 0
    for i, line in enumerate(lines):
        if i > lo:
            span = line.end_ms - lines[lo].start_ms
            gap = line.start_ms - lines[i - 1].end_ms
            full = chars + len(line.text) > max_chars or span > max_span_ms
            half = (
                chars * 2 >= max_chars
                or (lines[i - 1].end_ms - lines[lo].start_ms) * 2 >= max_span_ms
            )
            if full or (half and gap >= SOFT_GAP_MS):
                chunks.append((lo, i))
                lo, chars = i, 0
        chars += len(line.text)
    if lo < len(lines):
        chunks.append((lo, len(lines)))
    return chunks


def plan_scenes(
    lines: Sequence[TranscriptLine],
    shot_starts: Sequence[int],
    *,
    max_chars: int,
    max_span_ms: int,
) -> list[ScenePlan]:
    """Group `lines` into chunks and align the borders between chunks to `shot_starts`
    (ascending start times of the film's shots; the shots tile the film).

    The result partitions both lines and shots into the same number of consecutive,
    non-empty groups. Chunks whose border would fall on an already used shot border are
    merged into the previous scene. Empty `lines` or `shot_starts` give no scenes."""
    n_shots = len(shot_starts)
    if not lines or n_shots == 0:
        return []
    chunks = chunk_lines(lines, max_chars=max_chars, max_span_ms=max_span_ms)

    line_los = [chunks[0][0]]
    shot_los = [0]
    for lo, _hi in chunks[1:]:
        prev_end = lines[lo - 1].end_ms
        mid = (prev_end + max(prev_end, lines[lo].start_ms)) // 2
        k = _nearest_start(shot_starts, mid)
        if shot_los[-1] < k < n_shots:
            line_los.append(lo)
            shot_los.append(k)
    return [
        ScenePlan(
            line_lo=line_los[i],
            line_hi=line_los[i + 1] if i + 1 < len(line_los) else len(lines),
            shot_lo=shot_los[i],
            shot_hi=shot_los[i + 1] if i + 1 < len(shot_los) else n_shots,
        )
        for i in range(len(line_los))
    ]


def _nearest_start(shot_starts: Sequence[int], t_ms: int) -> int:
    """Index of the shot start closest to `t_ms` (the earlier one on a tie)."""
    i = bisect_left(shot_starts, t_ms)
    if i == 0:
        return 0
    if i == len(shot_starts):
        return len(shot_starts) - 1
    return i if shot_starts[i] - t_ms < t_ms - shot_starts[i - 1] else i - 1


def check_story_refs(
    act_scene_ids: Sequence[Sequence[str]],
    turning_point_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> list[str]:
    """Problems with the scene ids a story cites: unknown ids, or a scene in two acts."""
    valid = set(valid_ids)
    errors: list[str] = []
    seen: dict[str, int] = {}
    for n, ids in enumerate(act_scene_ids, 1):
        for sid in ids:
            if sid not in valid:
                errors.append(f"第 {n} 幕引用了不存在的场景 {sid}")
            elif sid in seen:
                errors.append(f"场景 {sid} 同时出现在第 {seen[sid]} 幕和第 {n} 幕")
            else:
                seen[sid] = n
    errors.extend(
        f"关键转折引用了不存在的场景 {sid}" for sid in turning_point_ids if sid not in valid
    )
    return errors
