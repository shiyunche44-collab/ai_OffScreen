"""Applying a person's edits to the segments of a plan. Pure.

Operations (`domain.plan_edit`) run in order on a copy; the result is checked as a whole, so a
request either gives a plan that compiles or is refused with the reason. Footage a person picks
or trims is locked (kept by rebuilds). Changing a voice does not speak anything here: the
segment is marked stale and keeps the old audio until the next build speaks it again."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from offscreen.algo.fitting import MAX_SPEED, MIN_SPEED
from offscreen.algo.original import original_segment
from offscreen.algo.plan_build import playback_ms
from offscreen.domain.index import Transcript
from offscreen.domain.plan import Clip, PlanSegment, VoiceSpec
from offscreen.domain.plan_edit import (
    AddClip,
    ClipSource,
    DeleteSegment,
    InsertOriginal,
    MoveSegment,
    PlanOp,
    RemoveClip,
    SetLocked,
    SetVoice,
    SwapClip,
    TrimClip,
)

COVER_TOLERANCE_MS = 42
"""A voice-over may outlast its footage by this much (one frame at 24 fps): the compiler holds
the last frame no longer than that."""
DEFAULT_CLIP_MS = 3000
"""Length of a clip added from a shot when there is no clip it replaces."""
INSERTED_PREFIX = "seg_o"
"""Ids of original-sound segments inserted by a person; the script's are `seg_01`, `seg_02`…"""


class PlanEditError(ValueError):
    pass


@dataclass(frozen=True)
class EditContext:
    asset_id: str
    shot_ranges: Mapping[str, tuple[int, int]]
    """shot id -> (start_ms, end_ms)."""
    asset_duration_ms: int | None = None
    transcript: Transcript | None = None
    """Needed to insert original sound."""


def apply_edits(
    segments: Sequence[PlanSegment], ops: Sequence[PlanOp], ctx: EditContext
) -> list[PlanSegment]:
    """The segments after `ops`. Raises PlanEditError naming the operation that failed."""
    if not ops:
        raise PlanEditError("no edits given")
    out = list(segments)
    for n, op in enumerate(ops, start=1):
        try:
            out = _apply(out, op, ctx)
        except PlanEditError as e:
            raise PlanEditError(f"edit {n} ({op.op}): {e}") from None
    _check(out, ctx)
    return out


def _apply(segs: list[PlanSegment], op: PlanOp, ctx: EditContext) -> list[PlanSegment]:
    out = list(segs)
    if isinstance(op, MoveSegment):
        seg = out.pop(_find(out, op.segment_id))
        if op.to_index > len(out):
            raise PlanEditError(f"position {op.to_index} is beyond the last segment")
        out.insert(op.to_index, seg)
        return out
    if isinstance(op, DeleteSegment):
        del out[_find(out, op.segment_id)]
        return out
    if isinstance(op, InsertOriginal):
        return _insert_original(out, op, ctx)

    i = _find(out, op.segment_id)
    seg = out[i]
    if isinstance(op, SetVoice):
        if seg.kind != "narration" or seg.voice is None:
            raise PlanEditError(f"{seg.id} is original sound: it has no voice")
        voice = VoiceSpec(
            voice_id=op.voice_id or seg.voice.voice_id, speed=op.speed or seg.voice.speed
        )
        out[i] = seg.model_copy(update={"voice": voice, "voice_pinned": True, "stale": True})
        return out

    clips = list(seg.clips)
    if isinstance(op, AddClip):
        at = len(clips) if op.index is None else op.index
        if at > len(clips):
            raise PlanEditError(f"{seg.id} has {len(clips)} clips: cannot insert at {at}")
        clips.insert(at, _clip(op.to, ctx, replacing=None))
    else:
        k = _clip_index(seg, op.index)
        if isinstance(op, SwapClip):
            clips[k] = _clip(op.to, ctx, replacing=clips[k])
        elif isinstance(op, RemoveClip):
            del clips[k]
        elif isinstance(op, TrimClip):
            if op.src_in_ms >= op.src_out_ms:
                raise PlanEditError("src_in_ms must be < src_out_ms")
            clips[k] = clips[k].model_copy(
                update={
                    "src_in_ms": op.src_in_ms,
                    "src_out_ms": op.src_out_ms,
                    "speed": op.speed or clips[k].speed,
                    "locked": True,
                }
            )
        elif isinstance(op, SetLocked):
            clips[k] = clips[k].model_copy(update={"locked": op.locked})
    out[i] = seg.model_copy(update={"clips": clips})
    return out


def _insert_original(
    segs: list[PlanSegment], op: InsertOriginal, ctx: EditContext
) -> list[PlanSegment]:
    if ctx.transcript is None:
        raise PlanEditError("the film has no transcript to take original sound from")
    taken = {s.id for s in segs}
    n = 1
    while f"{INSERTED_PREFIX}{n:02d}" in taken:
        n += 1
    try:
        seg = original_segment(
            f"{INSERTED_PREFIX}{n:02d}", op.line_refs, ctx.transcript, ctx.asset_duration_ms
        )
    except ValueError as e:
        raise PlanEditError(str(e)) from None
    at = 0 if op.after is None else _find(segs, op.after) + 1
    return [*segs[:at], seg, *segs[at:]]


def _clip(source: ClipSource, ctx: EditContext, replacing: Clip | None) -> Clip:
    if source.src_in_ms is not None and source.src_out_ms is not None:
        return Clip(
            asset_id=ctx.asset_id,
            shot_id=source.shot_id,
            src_in_ms=source.src_in_ms,
            src_out_ms=source.src_out_ms,
            speed=source.speed,
            locked=True,
        )
    assert source.shot_id is not None
    if source.shot_id not in ctx.shot_ranges:
        raise PlanEditError(f"unknown shot {source.shot_id}")
    start, end = ctx.shot_ranges[source.shot_id]
    available = end - start
    want = playback_ms(replacing) if replacing is not None else min(available, DEFAULT_CLIP_MS)
    if available >= want:
        length, speed = want, 1.0
    else:  # the shot is shorter than the clip it replaces: all of it, a little slower
        length, speed = available, max(MIN_SPEED, min(MAX_SPEED, available / want))
    src_in = start + (available - length) // 2
    return Clip(
        asset_id=ctx.asset_id,
        shot_id=source.shot_id,
        src_in_ms=src_in,
        src_out_ms=src_in + length,
        speed=round(speed, 6),
        locked=True,
    )


def _find(segs: Sequence[PlanSegment], segment_id: str) -> int:
    for i, s in enumerate(segs):
        if s.id == segment_id:
            return i
    raise PlanEditError(f"no segment {segment_id}")


def _clip_index(seg: PlanSegment, index: int) -> int:
    if index >= len(seg.clips):
        raise PlanEditError(f"{seg.id} has {len(seg.clips)} clips: no clip {index}")
    return index


def _check(segs: Sequence[PlanSegment], ctx: EditContext) -> None:
    if not segs:
        raise PlanEditError("a plan needs at least one segment")
    for seg in segs:
        if not seg.clips:
            if seg.kind == "narration" and seg.stale:
                continue
            raise PlanEditError(f"{seg.id} would have no footage")
        for c in seg.clips:
            if ctx.asset_duration_ms is not None and c.src_out_ms > ctx.asset_duration_ms:
                raise PlanEditError(
                    f"{seg.id}: a clip ends at {c.src_out_ms} ms, after the film "
                    f"({ctx.asset_duration_ms} ms)"
                )
        if seg.kind == "narration" and not seg.stale and seg.audio is not None:
            gap = seg.audio.duration_ms - sum(playback_ms(c) for c in seg.clips)
            if gap > COVER_TOLERANCE_MS:
                raise PlanEditError(
                    f"{seg.id}: footage is {gap} ms shorter than the voice-over; "
                    "add footage or lengthen a clip"
                )
