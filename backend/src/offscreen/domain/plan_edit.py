"""Edits of an EditPlan, as data: what a person asks for in the plan editor.

A request is a list of operations applied in order, all or nothing, to the current version and
stored as the next one (`algo.plan_edit.apply_edits`). Clips are addressed by their position in
the segment. Footage a person picks or trims is locked, so a rebuild keeps it."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import LineId, SegmentId, ShotId, Strict, TimeMs


class ClipSource(Strict):
    """Where footage comes from: a whole shot (the editor cuts the middle of it to the length
    of the clip it replaces), or any interval of the film."""

    shot_id: ShotId | None = None
    src_in_ms: TimeMs | None = None
    src_out_ms: TimeMs | None = None
    speed: float = Field(gt=0, default=1.0)

    @model_validator(mode="after")
    def _one_way_or_the_other(self) -> Self:
        interval = self.src_in_ms is not None or self.src_out_ms is not None
        if interval and (self.src_in_ms is None or self.src_out_ms is None):
            raise ValueError("an interval needs both src_in_ms and src_out_ms")
        if interval and self.src_in_ms is not None and self.src_out_ms is not None:
            if self.src_in_ms >= self.src_out_ms:
                raise ValueError("src_in_ms must be < src_out_ms")
        elif self.shot_id is None:
            raise ValueError("give a shot_id or an interval")
        return self


class SwapClip(Strict):
    op: Literal["swap_clip"] = "swap_clip"
    segment_id: SegmentId
    index: int = Field(ge=0)
    to: ClipSource


class AddClip(Strict):
    op: Literal["add_clip"] = "add_clip"
    segment_id: SegmentId
    to: ClipSource
    index: int | None = Field(default=None, ge=0, description="Where to put it; null: the end.")


class RemoveClip(Strict):
    op: Literal["remove_clip"] = "remove_clip"
    segment_id: SegmentId
    index: int = Field(ge=0)


class TrimClip(Strict):
    op: Literal["trim_clip"] = "trim_clip"
    segment_id: SegmentId
    index: int = Field(ge=0)
    src_in_ms: TimeMs
    src_out_ms: TimeMs
    speed: float | None = Field(default=None, gt=0, description="null: keep the speed.")


class SetLocked(Strict):
    op: Literal["set_locked"] = "set_locked"
    segment_id: SegmentId
    index: int = Field(ge=0)
    locked: bool


class MoveSegment(Strict):
    op: Literal["move_segment"] = "move_segment"
    segment_id: SegmentId
    to_index: int = Field(ge=0)


class DeleteSegment(Strict):
    op: Literal["delete_segment"] = "delete_segment"
    segment_id: SegmentId


class InsertOriginal(Strict):
    op: Literal["insert_original"] = "insert_original"
    line_refs: list[LineId] = Field(min_length=1)
    after: SegmentId | None = Field(default=None, description="null: at the start.")


class SetVoice(Strict):
    """A new voice and / or speed for a narration segment. It is spoken again by the next plan
    build (the segment is marked stale until then) and keeps this voice through rebuilds."""

    op: Literal["set_voice"] = "set_voice"
    segment_id: SegmentId
    voice_id: str | None = Field(default=None, min_length=1)
    speed: float | None = Field(default=None, ge=0.5, le=2.0)

    @model_validator(mode="after")
    def _something(self) -> Self:
        if self.voice_id is None and self.speed is None:
            raise ValueError("give a voice_id, a speed, or both")
        return self


PlanOp = Annotated[
    SwapClip
    | AddClip
    | RemoveClip
    | TrimClip
    | SetLocked
    | MoveSegment
    | DeleteSegment
    | InsertOriginal
    | SetVoice,
    Field(discriminator="op"),
]
