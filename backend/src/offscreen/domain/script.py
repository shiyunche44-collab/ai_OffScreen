"""L2 Script: segmented commentary text. Versioned, immutable documents."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import (
    LineId,
    ProjectId,
    SceneId,
    ScriptId,
    SegmentId,
    Strict,
    Versioned,
)

Author = Literal["ai", "human"]


class ScriptParams(Strict):
    style: str
    target_duration_s: int = Field(gt=0)
    perspective: Literal["first", "third"] = "third"
    spoil_ending: bool = True
    language: str = "zh"
    voice_id: str


class OutlineBeat(Strict):
    beat: str
    scene_refs: list[SceneId] = Field(min_length=1)
    target_s: int = Field(gt=0)


class ScriptSegment(Strict):
    id: SegmentId
    kind: Literal["narration", "original"]
    beat: str | None = None
    text: str
    scene_refs: list[SceneId] = []
    line_refs: list[LineId] = []

    @model_validator(mode="after")
    def _kind_rules(self) -> Self:
        if self.kind == "narration":
            if not self.text.strip():
                raise ValueError("narration segment needs text")
            if not self.scene_refs:
                raise ValueError("narration segment needs scene_refs (fact anchor)")
        elif not self.line_refs:
            raise ValueError("original segment needs line_refs")
        return self


class Annotation(Strict):
    segment_id: SegmentId
    type: Literal["fact_check", "rule", "style"]
    message: str


class Script(Versioned):
    id: ScriptId
    project_id: ProjectId
    version: int = Field(ge=1)
    parent_version: int | None = None
    author: Author
    params: ScriptParams
    outline: list[OutlineBeat] = []
    segments: list[ScriptSegment]
    annotations: list[Annotation] = []

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment ids")
        unknown = {a.segment_id for a in self.annotations} - set(ids)
        if unknown:
            raise ValueError(f"annotations reference unknown segments: {sorted(unknown)}")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent_version must be < version")
        return self
