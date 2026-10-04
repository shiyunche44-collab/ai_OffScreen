"""compile(plan) -> Timeline: the deterministic step between editing and rendering. Pure.

Everything lands on the output frame grid (ARCHITECTURE §7.4). Positions are rounded from
*cumulative* time, never per item, so rounding error does not build up across segments or
clips: segment k ends at `round(sum of durations up to k * fps)`, however many there are.

Conventions the renderer relies on:
- The video track tiles `[0, duration_frames)`. A video item shows exactly `f1 - f0` frames
  starting at `src_in_ms` (at `speed`); `src_out_ms` is the plan's, and is never extended.
- A segment lasts as long as its voice-over (an original-sound segment: as long as its
  clips). Clips that would start after that are dropped; clips that fall short by more than
  one frame are an error, because the renderer cannot invent footage.
- `NarrationItem.file` is the plan's `AudioRef.file`, relative to the plan's directory.
- Source-audio items follow the clips (one per video item). `mode="mute"` emits none;
  `duck` and `full` both play at `gain_db`: ducking under narration is the mixer's job.
- Subtitles come from the segment's text snapshot and the voice-over's character timings."""

from __future__ import annotations

from offscreen.algo.captions import split_caption_lines
from offscreen.domain.common import Rational
from offscreen.domain.plan import Clip, EditPlan, PlanSegment
from offscreen.domain.timeline import (
    NarrationItem,
    OutputSpec,
    PlanRef,
    SourceAudioItem,
    SubtitleItem,
    Timeline,
    VideoItem,
)


class CompileError(ValueError):
    pass


def _clip_ms(clip: Clip) -> float:
    return (clip.src_out_ms - clip.src_in_ms) / clip.speed


def _segment_ms(seg: PlanSegment) -> int:
    if seg.audio is not None:
        return seg.audio.duration_ms
    return round(sum(_clip_ms(c) for c in seg.clips))


def compile_timeline(plan: EditPlan, output: OutputSpec, *, subtitle_max_chars: int) -> Timeline:
    if not plan.segments:
        raise CompileError("plan has no segments")
    fps = output.fps
    one_frame_ms = 1000 * fps.den / fps.num

    video: list[VideoItem] = []
    narration: list[NarrationItem] = []
    source_audio: list[SourceAudioItem] = []
    subtitles: list[SubtitleItem] = []

    cum_ms = 0
    f_cursor = 0
    for seg in plan.segments:
        if seg.stale:
            raise CompileError(f"{seg.id} is stale: rebuild the plan first")
        seg_ms = _segment_ms(seg)
        cum_ms += seg_ms
        seg_f0 = f_cursor
        seg_f1 = max(seg_f0 + 1, fps.frames_for_ms(cum_ms))

        items = _place_clips(seg, seg_ms, cum_ms - seg_ms, seg_f0, seg_f1, fps, one_frame_ms)
        video.extend(items)
        if seg.source_audio.mode != "mute":
            source_audio.extend(
                SourceAudioItem(
                    seg=seg.id,
                    f0=v.f0,
                    f1=v.f1,
                    asset_id=v.asset_id,
                    stem=seg.source_audio.stem,
                    src_in_ms=v.src_in_ms,
                    gain_db=seg.source_audio.gain_db,
                )
                for v in items
            )
        if seg.audio is not None:
            narration.append(NarrationItem(seg=seg.id, f0=seg_f0, file=seg.audio.file))
            if seg.text:
                subtitles.extend(_subtitles(seg, seg_f0, seg_f1, fps, subtitle_max_chars))
        f_cursor = seg_f1

    return Timeline(
        plan_ref=PlanRef(id=plan.id, version=plan.version),
        output=output,
        duration_frames=f_cursor,
        video=video,
        narration=narration,
        source_audio=source_audio,
        subtitles=subtitles,
    )


def _place_clips(
    seg: PlanSegment,
    seg_ms: int,
    base_ms: int,
    f0: int,
    f1: int,
    fps: Rational,
    one_frame_ms: float,
) -> list[VideoItem]:
    """`base_ms` is the programme time at which the segment starts: clip borders are rounded
    from time since the very start, so a clip is off by at most one frame in length."""
    kept: list[Clip] = []
    before = 0.0
    for clip in seg.clips:
        if kept and before >= seg_ms - one_frame_ms / 2:
            break  # the segment is already covered
        kept.append(clip)
        before += _clip_ms(clip)
    if before < seg_ms - one_frame_ms:
        raise CompileError(
            f"{seg.id}: clips cover {before:.0f} ms but the segment lasts {seg_ms} ms"
        )
    kept = kept[: f1 - f0]  # a one-frame-per-clip floor: never more clips than frames

    out: list[VideoItem] = []
    cum = 0.0
    start = f0
    for i, clip in enumerate(kept):
        cum += _clip_ms(clip)
        room_after = len(kept) - 1 - i
        if room_after == 0:
            end = f1
        else:
            end = min(max(fps.frames_for_ms(base_ms + round(cum)), start + 1), f1 - room_after)
        out.append(
            VideoItem(
                seg=seg.id,
                f0=start,
                f1=end,
                asset_id=clip.asset_id,
                src_in_ms=clip.src_in_ms,
                src_out_ms=clip.src_out_ms,
                speed=clip.speed,
            )
        )
        start = end
    return out


def _subtitles(
    seg: PlanSegment, seg_f0: int, seg_f1: int, fps: Rational, max_chars: int
) -> list[SubtitleItem]:
    assert seg.audio is not None and seg.text is not None
    lines = split_caption_lines(seg.text, seg.audio.char_timings, seg.audio.duration_ms, max_chars)
    out: list[SubtitleItem] = []
    cursor = seg_f0
    for line in lines:
        f0 = max(cursor, seg_f0 + fps.frames_for_ms(line.start_ms))
        f1 = min(seg_f1, max(f0 + 1, seg_f0 + fps.frames_for_ms(line.end_ms)))
        if f0 >= seg_f1:
            break
        out.append(SubtitleItem(f0=f0, f1=f1, text=line.text))
        cursor = f1
    return out
