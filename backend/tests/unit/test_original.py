"""Unit and property tests for original-sound segments."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.original import PAD_MS, find_overlaps, original_clip, original_segment
from offscreen.domain.index import Transcript, TranscriptLine
from offscreen.domain.plan import AudioRef, Clip, PlanSegment, VoiceSpec

ASSET = "ast_test"


def _transcript(*spans: tuple[int, int]) -> Transcript:
    return Transcript(
        asset_id=ASSET,
        language="en",
        source="test",
        lines=[
            TranscriptLine(id=f"ln_{i:04d}", start_ms=a, end_ms=b, text="x")
            for i, (a, b) in enumerate(spans)
        ],
    )


def _narration(seg_id: str, a: int, b: int, asset: str = ASSET) -> PlanSegment:
    return PlanSegment(
        id=seg_id,
        kind="narration",
        text="t",
        voice=VoiceSpec(voice_id="v"),
        audio=AudioRef(file="a.wav", duration_ms=1000),
        clips=[Clip(asset_id=asset, src_in_ms=a, src_out_ms=b)],
    )


def test_pads_200ms_each_side() -> None:
    clip = original_clip(["ln_0000"], _transcript((5000, 7000)))
    assert (clip.src_in_ms, clip.src_out_ms) == (5000 - PAD_MS, 7000 + PAD_MS)
    assert clip.locked and clip.speed == 1.0 and clip.asset_id == ASSET


def test_multiple_lines_span_from_first_to_last() -> None:
    clip = original_clip(
        ["ln_0002", "ln_0000"], _transcript((1000, 2000), (3000, 4000), (5000, 6000))
    )
    assert (clip.src_in_ms, clip.src_out_ms) == (800, 6200)


def test_clamped_to_asset_bounds() -> None:
    clip = original_clip(["ln_0000"], _transcript((100, 900)), asset_duration_ms=1000)
    assert (clip.src_in_ms, clip.src_out_ms) == (0, 1000)


def test_unknown_or_empty_refs_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        original_clip(["ln_9999"], _transcript((0, 1000)))
    with pytest.raises(ValueError, match="needs line_refs"):
        original_clip([], _transcript((0, 1000)))


def test_line_beyond_asset_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        original_clip(["ln_0000"], _transcript((5000, 6000)), asset_duration_ms=1000)


def test_original_segment_plays_full_mix() -> None:
    seg = original_segment("seg_02", ["ln_0000"], _transcript((1000, 2000)))
    assert seg.kind == "original" and seg.audio is None
    assert (seg.source_audio.mode, seg.source_audio.stem, seg.source_audio.gain_db) == (
        "full",
        "mix",
        0.0,
    )


def test_overlap_detected_and_adjacent_allowed() -> None:
    orig = original_segment("seg_02", ["ln_0000"], _transcript((5000, 7000)))  # 4800–7200
    assert find_overlaps([_narration("seg_01", 4000, 4800), orig]) == []
    hits = find_overlaps([_narration("seg_01", 4000, 5000), orig])
    assert [(h.original_id, h.narration_id, h.start_ms, h.end_ms) for h in hits] == [
        ("seg_02", "seg_01", 4800, 5000)
    ]


def test_other_asset_does_not_overlap() -> None:
    orig = original_segment("seg_02", ["ln_0000"], _transcript((5000, 7000)))
    assert find_overlaps([_narration("seg_01", 0, 10000, asset="ast_other"), orig]) == []


@given(
    start=st.integers(0, 10_000_000),
    length=st.integers(1, 60_000),
    duration=st.none() | st.integers(1, 20_000_000),
)
def test_clip_always_covers_line_within_bounds(
    start: int, length: int, duration: int | None
) -> None:
    end = start + length
    tr = _transcript((start, end))
    if duration is not None and max(0, start - PAD_MS) >= duration:
        with pytest.raises(ValueError):
            original_clip(["ln_0000"], tr, duration)
        return
    clip = original_clip(["ln_0000"], tr, duration)
    assert clip.src_in_ms >= 0 and clip.src_in_ms < clip.src_out_ms
    assert clip.src_in_ms == max(0, start - PAD_MS)
    cap = end + PAD_MS if duration is None else min(end + PAD_MS, duration)
    assert clip.src_out_ms == cap
    assert clip.src_in_ms <= start
