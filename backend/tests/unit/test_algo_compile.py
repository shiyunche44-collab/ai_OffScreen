from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.compile import CompileError, compile_timeline
from offscreen.domain.common import Rational
from offscreen.domain.plan import (
    AudioRef,
    Clip,
    EditPlan,
    PlanSegment,
    ScriptRef,
    SourceAudio,
    VoiceSpec,
)
from offscreen.domain.timeline import OutputSpec

FPS_CHOICES = [
    Rational(num=24000, den=1001),
    Rational(num=24, den=1),
    Rational(num=25, den=1),
    Rational(num=30, den=1),
    Rational(num=60, den=1),
]


FPS25 = Rational(num=25, den=1)


def spec(fps: Rational = FPS25) -> OutputSpec:
    return OutputSpec(profile="source", width=1920, height=800, fps=fps, layout="keep")


def seg(
    i: int,
    clip_ms: list[int],
    audio_ms: int | None = None,
    text: str | None = "你好世界",
    gain: float = -20.0,
    mode: str = "duck",
) -> PlanSegment:
    audio_ms = sum(clip_ms) if audio_ms is None else audio_ms
    clips, t = [], 1000 * i
    for d in clip_ms:
        clips.append(Clip(asset_id="ast_t", shot_id=f"sh_{t}", src_in_ms=t, src_out_ms=t + d))
        t += d + 50
    return PlanSegment(
        id=f"seg_{i:02d}",
        kind="narration",
        text=text,
        voice=VoiceSpec(voice_id="v"),
        audio=AudioRef(file=f"tts/{i}.mp3", duration_ms=audio_ms),
        clips=clips,
        source_audio=SourceAudio(mode=mode, stem="mix", gain_db=gain),  # type: ignore[arg-type]
    )


def plan(*segs: PlanSegment) -> EditPlan:
    return EditPlan(
        id="pln_t",
        project_id="prj_t",
        version=3,
        author="ai",
        script_ref=ScriptRef(id="scr_t", version=1),
        segments=list(segs),
    )


def test_tracks_for_a_simple_plan() -> None:
    p = plan(seg(1, [2000, 1000], text="你好，世界。"), seg(2, [1500]))
    tl = compile_timeline(p, spec(), subtitle_max_chars=22)

    assert tl.plan_ref.id == "pln_t" and tl.plan_ref.version == 3
    assert tl.duration_frames == 113  # 4.5 s at 25 fps = 112.5 frames, rounded half up
    assert [(v.seg, v.f0, v.f1) for v in tl.video] == [
        ("seg_01", 0, 50),
        ("seg_01", 50, 75),
        ("seg_02", 75, 113),
    ]
    assert tl.video[0].src_in_ms == 1000 and tl.video[0].speed == 1.0
    assert [(n.seg, n.f0, n.file) for n in tl.narration] == [
        ("seg_01", 0, "tts/1.mp3"),
        ("seg_02", 75, "tts/2.mp3"),
    ]
    assert [(s.f0, s.f1, s.gain_db, s.stem) for s in tl.source_audio] == [
        (v.f0, v.f1, -20.0, "mix") for v in tl.video
    ]
    assert tl.subtitles and all(0 <= s.f0 < s.f1 <= tl.duration_frames for s in tl.subtitles)
    assert [s.text for s in tl.subtitles][:2] == ["你好，世界", "你好世界"]


def test_mute_emits_no_source_audio() -> None:
    tl = compile_timeline(plan(seg(1, [1000], mode="mute")), spec(), subtitle_max_chars=22)
    assert tl.source_audio == []


def test_extra_clips_are_dropped_and_shortfall_is_an_error() -> None:
    p = plan(seg(1, [1000, 1000, 1000], audio_ms=1500))
    tl = compile_timeline(p, spec(), subtitle_max_chars=22)
    assert [(v.f0, v.f1) for v in tl.video] == [
        (0, 25),
        (25, 38),
    ]  # third clip starts after the end

    with pytest.raises(CompileError, match="clips cover"):
        compile_timeline(plan(seg(1, [1000], audio_ms=1500)), spec(), subtitle_max_chars=22)


def test_stale_and_empty_plans_are_rejected() -> None:
    stale = PlanSegment(id="seg_01", kind="narration", stale=True, voice=VoiceSpec(voice_id="v"))
    with pytest.raises(CompileError, match="stale"):
        compile_timeline(plan(stale), spec(), subtitle_max_chars=22)
    with pytest.raises(CompileError, match="no segments"):
        compile_timeline(plan(), spec(), subtitle_max_chars=22)


def test_tiny_segments_still_get_a_frame_each() -> None:
    p = plan(seg(1, [10]), seg(2, [10]), seg(3, [10]))
    tl = compile_timeline(p, spec(), subtitle_max_chars=22)
    assert [(v.f0, v.f1) for v in tl.video] == [(0, 1), (1, 2), (2, 3)]
    assert tl.duration_frames == 3


def test_rounding_does_not_accumulate_across_segments() -> None:
    # 100 segments of 60 ms at 24 fps are 1.44 frames each. Rounding every segment on its
    # own would give 100 frames (4.2 s of programme lost); rounding cumulative time gives 144.
    p = plan(*[seg(i, [60]) for i in range(1, 101)])
    tl = compile_timeline(p, spec(Rational(num=24, den=1)), subtitle_max_chars=22)
    assert tl.duration_frames == 144
    assert {v.f1 - v.f0 for v in tl.video} == {1, 2}


durations = st.lists(st.integers(120, 5000), min_size=1, max_size=4)


@given(st.lists(durations, min_size=1, max_size=12), st.sampled_from(FPS_CHOICES))
def test_frame_grid_properties(segs_clips: list[list[int]], fps: Rational) -> None:
    p = plan(*[seg(i, c, text="你好世界。再见。") for i, c in enumerate(segs_clips, 1)])
    tl = compile_timeline(p, spec(fps), subtitle_max_chars=14)  # Timeline validates the tiling
    total_ms = sum(sum(c) for c in segs_clips)

    # Total frames = the segments' frames added up = the whole programme rounded once.
    per_segment: dict[str, int] = {}
    for v in tl.video:
        per_segment[v.seg] = per_segment.get(v.seg, 0) + v.f1 - v.f0
    assert sum(per_segment.values()) == tl.duration_frames == fps.frames_for_ms(total_ms)

    # Each segment ends where the cumulative time says it does, so nothing drifts.
    cum, end = 0, 0
    for i, c in enumerate(segs_clips, 1):
        cum += sum(c)
        end += per_segment[f"seg_{i:02d}"]
        assert end == fps.frames_for_ms(cum)

    # Every clip lasts what the plan says, to within a frame.
    one_frame = 1000 * fps.den / fps.num
    for v in tl.video:
        shown = (v.f1 - v.f0) * one_frame
        assert abs(shown - (v.src_out_ms - v.src_in_ms) / v.speed) <= one_frame + 1e-6

    assert all(s.f1 <= tl.duration_frames for s in tl.subtitles + tl.source_audio)  # type: ignore[operator]
    assert all(a.f1 <= b.f0 for a, b in zip(tl.subtitles, tl.subtitles[1:], strict=False))
