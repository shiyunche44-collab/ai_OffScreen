"""The plan editor's operations: each one, and the checks on the plan they leave behind."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from offscreen.algo.plan_build import playback_ms, text_digest
from offscreen.algo.plan_edit import EditContext, PlanEditError, apply_edits
from offscreen.domain.index import Transcript, TranscriptLine
from offscreen.domain.plan import AudioRef, Clip, PlanSegment, VoiceSpec
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

ASSET = "ast_t1"
SHOTS = {f"sh_{i:04d}": (i * 3000, (i + 1) * 3000) for i in range(10)}
TRANSCRIPT = Transcript(
    asset_id=ASSET,
    language="zh",
    source="test",
    lines=[
        TranscriptLine(id="ln_0001", start_ms=20_000, end_ms=22_000, text="你是谁"),
        TranscriptLine(id="ln_0002", start_ms=25_000, end_ms=26_000, text="我是龙"),
    ],
)
CTX = EditContext(ASSET, SHOTS, asset_duration_ms=30_000, transcript=TRANSCRIPT)


def clip(a: int, b: int, shot: str | None = None, *, locked: bool = False) -> Clip:
    return Clip(asset_id=ASSET, shot_id=shot, src_in_ms=a, src_out_ms=b, locked=locked)


def narration(i: int, ms: int, clips: list[Clip]) -> PlanSegment:
    text = f"第{i}段的文案"
    return PlanSegment(
        id=f"seg_{i:02d}",
        kind="narration",
        text=text,
        text_hash=text_digest(text),
        voice=VoiceSpec(voice_id="v", speed=1.0),
        audio=AudioRef(file=f"tts/{i}.wav", duration_ms=ms),
        clips=clips,
    )


def plan() -> list[PlanSegment]:
    return [
        narration(1, 4000, [clip(0, 2000, "sh_0000"), clip(3000, 5000, "sh_0001")]),
        narration(2, 3000, [clip(12_000, 15_000, "sh_0004")]),
        narration(3, 3000, [clip(15_000, 18_000, "sh_0005")]),
    ]


def run(*ops: PlanOp) -> list[PlanSegment]:
    return apply_edits(plan(), list(ops), CTX)


def test_swapping_a_clip_for_a_shot_keeps_its_length_and_locks_the_new_one() -> None:
    out = run(SwapClip(segment_id="seg_01", index=1, to=ClipSource(shot_id="sh_0007")))

    old, new = plan()[0].clips[1], out[0].clips[1]
    assert new.shot_id == "sh_0007" and new.locked and old.locked is False
    assert playback_ms(new) == playback_ms(old) == 2000
    assert new.src_in_ms >= 21_000 and new.src_out_ms <= 24_000  # inside the shot
    assert out[0].clips[0] == plan()[0].clips[0]


def test_swapping_for_a_shorter_shot_uses_all_of_it_a_little_slower() -> None:
    ctx = EditContext(ASSET, {**SHOTS, "sh_9999": (0, 1800)}, asset_duration_ms=30_000)

    out = apply_edits(
        plan(), [SwapClip(segment_id="seg_01", index=1, to=ClipSource(shot_id="sh_9999"))], ctx
    )

    c = out[0].clips[1]
    assert (c.src_in_ms, c.src_out_ms) == (0, 1800) and c.speed == pytest.approx(0.9)


def test_swapping_for_any_interval_takes_it_as_given() -> None:
    out = run(
        SwapClip(
            segment_id="seg_02", index=0, to=ClipSource(src_in_ms=1000, src_out_ms=4500, speed=1.0)
        )
    )

    assert (out[1].clips[0].src_in_ms, out[1].clips[0].src_out_ms) == (1000, 4500)
    assert out[1].clips[0].locked


def test_trim_locks_and_may_change_the_speed() -> None:
    out = run(
        TrimClip(segment_id="seg_02", index=0, src_in_ms=12_000, src_out_ms=15_200, speed=1.05)
    )

    c = out[1].clips[0]
    assert (c.src_in_ms, c.src_out_ms, c.speed, c.locked) == (12_000, 15_200, 1.05, True)
    assert c.shot_id == "sh_0004"


def test_lock_and_unlock() -> None:
    locked = run(SetLocked(segment_id="seg_02", index=0, locked=True))
    assert locked[1].clips[0].locked
    again = apply_edits(locked, [SetLocked(segment_id="seg_02", index=0, locked=False)], CTX)
    assert again[1].clips[0].locked is False


def test_add_and_remove_clips() -> None:
    out = run(
        AddClip(segment_id="seg_02", to=ClipSource(shot_id="sh_0008"), index=0),
        RemoveClip(segment_id="seg_02", index=1),
    )

    assert [c.shot_id for c in out[1].clips] == ["sh_0008"]
    assert out[1].clips[0].locked


def test_segments_move_and_are_deleted() -> None:
    out = run(
        MoveSegment(segment_id="seg_03", to_index=0),
        DeleteSegment(segment_id="seg_02"),
    )

    assert [s.id for s in out] == ["seg_03", "seg_01"]


def test_original_sound_is_inserted_after_a_segment_with_a_fresh_id() -> None:
    out = run(
        InsertOriginal(line_refs=["ln_0001"], after="seg_01"),
        InsertOriginal(line_refs=["ln_0002"], after=None),
    )

    assert [s.id for s in out] == ["seg_o02", "seg_01", "seg_o01", "seg_02", "seg_03"]
    inserted = out[2]
    assert inserted.kind == "original" and inserted.line_refs == ["ln_0001"]
    assert (inserted.clips[0].src_in_ms, inserted.clips[0].src_out_ms) == (19_800, 22_200)


def test_a_new_voice_marks_the_segment_stale_and_pinned() -> None:
    out = run(SetVoice(segment_id="seg_02", voice_id="w"), SetVoice(segment_id="seg_02", speed=1.2))

    seg = out[1]
    assert seg.voice == VoiceSpec(voice_id="w", speed=1.2)
    assert seg.stale and seg.voice_pinned
    assert seg.audio == plan()[1].audio  # the old audio stays until the next build
    assert not out[0].stale


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (DeleteSegment(segment_id="seg_09"), "no segment seg_09"),
        (SwapClip(segment_id="seg_01", index=5, to=ClipSource(shot_id="sh_0001")), "no clip 5"),
        (SwapClip(segment_id="seg_01", index=0, to=ClipSource(shot_id="sh_9")), "unknown shot"),
        (AddClip(segment_id="seg_01", to=ClipSource(shot_id="sh_0001"), index=9), "insert at 9"),
        (MoveSegment(segment_id="seg_01", to_index=9), "beyond the last"),
        (TrimClip(segment_id="seg_01", index=0, src_in_ms=500, src_out_ms=500), "src_in_ms"),
        (InsertOriginal(line_refs=["ln_404"]), "ln_404"),
        (InsertOriginal(line_refs=["ln_0001"], after="seg_09"), "no segment seg_09"),
    ],
)
def test_bad_operations_are_refused_with_the_operation_named(op: PlanOp, message: str) -> None:
    with pytest.raises(PlanEditError, match=message) as e:
        run(op)
    assert "edit 1" in str(e.value)


def test_footage_shorter_than_the_voice_over_is_refused() -> None:
    with pytest.raises(PlanEditError, match=r"seg_01: footage is 2000 ms shorter"):
        run(RemoveClip(segment_id="seg_01", index=1))
    with pytest.raises(PlanEditError, match="would have no footage"):
        run(RemoveClip(segment_id="seg_02", index=0))


def test_a_stale_segment_may_lose_its_footage_while_it_waits_for_the_next_build() -> None:
    out = run(SetVoice(segment_id="seg_02", speed=1.5), RemoveClip(segment_id="seg_02", index=0))

    assert out[1].clips == [] and out[1].stale


def test_clips_must_lie_inside_the_film() -> None:
    with pytest.raises(PlanEditError, match="after the film"):
        run(
            SwapClip(
                segment_id="seg_02", index=0, to=ClipSource(src_in_ms=28_000, src_out_ms=31_000)
            )
        )


def test_an_empty_plan_is_refused() -> None:
    with pytest.raises(PlanEditError, match="at least one segment"):
        run(
            DeleteSegment(segment_id="seg_01"),
            DeleteSegment(segment_id="seg_02"),
            DeleteSegment(segment_id="seg_03"),
        )


def test_one_bad_operation_spoils_the_whole_request() -> None:
    source = plan()
    with pytest.raises(PlanEditError, match="edit 2"):
        apply_edits(
            source,
            [MoveSegment(segment_id="seg_03", to_index=0), DeleteSegment(segment_id="seg_09")],
            CTX,
        )
    assert [s.id for s in source] == ["seg_01", "seg_02", "seg_03"]  # input untouched


def test_inserting_original_sound_needs_the_transcript() -> None:
    ctx = EditContext(ASSET, SHOTS, asset_duration_ms=30_000)

    with pytest.raises(PlanEditError, match="transcript"):
        apply_edits(plan(), [InsertOriginal(line_refs=["ln_0001"])], ctx)


def test_no_edits_is_refused() -> None:
    with pytest.raises(PlanEditError, match="no edits"):
        apply_edits(plan(), [], CTX)


class TestOperationSchema:
    ops = TypeAdapter(list[PlanOp])

    def test_requests_are_parsed_by_their_op_name(self) -> None:
        parsed = self.ops.validate_python(
            [
                {"op": "swap_clip", "segment_id": "seg_01", "index": 0, "to": {"shot_id": "sh_1"}},
                {"op": "set_voice", "segment_id": "seg_01", "speed": 1.2},
                {"op": "insert_original", "line_refs": ["ln_1"]},
            ]
        )

        assert [type(o).__name__ for o in parsed] == ["SwapClip", "SetVoice", "InsertOriginal"]

    @pytest.mark.parametrize(
        "bad",
        [
            {"op": "swap_clip", "segment_id": "seg_01", "index": 0, "to": {}},
            {"op": "swap_clip", "segment_id": "seg_01", "index": 0, "to": {"src_in_ms": 5}},
            {
                "op": "swap_clip",
                "segment_id": "seg_01",
                "index": 0,
                "to": {"src_in_ms": 9, "src_out_ms": 5},
            },
            {"op": "set_voice", "segment_id": "seg_01"},
            {"op": "set_voice", "segment_id": "seg_01", "speed": 9},
            {"op": "insert_original", "line_refs": []},
            {"op": "explode", "segment_id": "seg_01"},
            {"op": "delete_segment", "segment_id": "seg_01", "extra": 1},
        ],
    )
    def test_malformed_requests_are_rejected(self, bad: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            self.ops.validate_python([bad])
