"""L2 Script: segmented commentary text. Versioned, immutable documents."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import (
    AssetId,
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
    focus: str = ""
    """What this part tells, in a sentence: the instruction the writing step follows."""


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


class ScriptContent(Strict):
    """What a person (or the writing step) supplies to save a new version: everything of a
    `Script` except the identity and history, which the document store assigns."""

    params: ScriptParams
    outline: list[OutlineBeat] = []
    segments: list[ScriptSegment]
    annotations: list[Annotation] = []

    @classmethod
    def of(cls, script: Script) -> ScriptContent:
        return cls(
            params=script.params,
            outline=script.outline,
            segments=script.segments,
            annotations=script.annotations,
        )


class ScriptOutline(Versioned):
    """The plan of a script before it is written: which beats, which scenes each draws on, how
    many seconds each gets. A person can edit it before the text is written (ARCHITECTURE §7.2)."""

    asset_id: AssetId
    style: str
    target_duration_s: int = Field(gt=0)
    beats: list[OutlineBeat] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_beats(self) -> Self:
        names = [b.beat for b in self.beats]
        if any(not n.strip() for n in names):
            raise ValueError("beat names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("beat names must be unique")
        return self

    @property
    def total_s(self) -> int:
        return sum(b.target_s for b in self.beats)


class ScriptReview(Versioned):
    """What the fact checker found in a script: annotations only, the text is never changed."""

    asset_id: AssetId
    annotations: list[Annotation] = []
