"""Original-sound segments: turn a script segment's `line_refs` into a source clip.

An original segment (`kind=original`) plays the film's own audio and picture for the
referenced lines, with 200 ms of lead-in and tail. It must not reuse footage that a
narration segment already shows (same asset, overlapping source range).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.domain.index import Transcript
from offscreen.domain.plan import Clip, PlanSegment, SourceAudio

PAD_MS = 200


def original_clip(
    line_refs: Sequence[str],
    transcript: Transcript,
    asset_duration_ms: int | None = None,
) -> Clip:
    """One locked clip spanning every referenced line, padded by `PAD_MS` on both sides.

    The span is clamped to the asset ([0, asset_duration_ms]) when its length is known.
    Raises ValueError for empty or unknown line refs.
    """
    if not line_refs:
        raise ValueError("original segment needs line_refs")
    by_id = {ln.id: ln for ln in transcript.lines}
    missing = [r for r in line_refs if r not in by_id]
    if missing:
        raise ValueError(f"unknown line refs: {', '.join(missing)}")
    lines = [by_id[r] for r in line_refs]
    start = max(0, min(ln.start_ms for ln in lines) - PAD_MS)
    end = max(ln.end_ms for ln in lines) + PAD_MS
    if asset_duration_ms is not None:
        end = min(end, asset_duration_ms)
    if start >= end:
        raise ValueError("referenced lines lie outside the asset")
    return Clip(asset_id=transcript.asset_id, src_in_ms=start, src_out_ms=end, locked=True)


def original_segment(
    segment_id: str,
    line_refs: Sequence[str],
    transcript: Transcript,
    asset_duration_ms: int | None = None,
) -> PlanSegment:
    """EditPlan segment for an original-sound script segment (full mix, no gain)."""
    return PlanSegment(
        id=segment_id,
        kind="original",
        clips=[original_clip(line_refs, transcript, asset_duration_ms)],
        source_audio=SourceAudio(mode="full", stem="mix", gain_db=0.0),
    )


@dataclass(frozen=True)
class Overlap:
    """An original segment whose source range intersects a narration segment's clip."""

    original_id: str
    narration_id: str
    asset_id: str
    start_ms: int
    end_ms: int


def find_overlaps(segments: Sequence[PlanSegment]) -> list[Overlap]:
    """Source-range overlaps (half-open) between original clips and narration clips."""
    found: list[Overlap] = []
    for orig in segments:
        if orig.kind != "original":
            continue
        for narr in segments:
            if narr.kind != "narration":
                continue
            for oc in orig.clips:
                for nc in narr.clips:
                    if oc.asset_id != nc.asset_id:
                        continue
                    start = max(oc.src_in_ms, nc.src_in_ms)
                    end = min(oc.src_out_ms, nc.src_out_ms)
                    if start < end:
                        found.append(Overlap(orig.id, narr.id, oc.asset_id, start, end))
    return found
