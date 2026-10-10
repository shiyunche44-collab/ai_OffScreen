"""L3 EditPlan: per segment, the voice-over audio + chosen clips + audio policy."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import (
    AssetId,
    LineId,
    PlanId,
    ProjectId,
    ScriptId,
    SegmentId,
    Sha256,
    ShotId,
    Strict,
    TimeMs,
    Versioned,
)
from offscreen.domain.script import Author


class ScriptRef(Strict):
    id: ScriptId
    version: int = Field(ge=1)


class VoiceSpec(Strict):
    voice_id: str
    speed: float = Field(gt=0, default=1.0)


class AudioRef(Strict):
    file: str  # relative to the directory holding the plan document (like Shot.keyframes)
    duration_ms: int = Field(gt=0)
    # per-character (start_ms, end_ms) inside the audio, if the TTS engine provides them
    char_timings: list[tuple[TimeMs, TimeMs]] = []


class Clip(Strict):
    asset_id: AssetId
    shot_id: ShotId | None = None
    src_in_ms: TimeMs
    src_out_ms: TimeMs
    speed: float = Field(gt=0, default=1.0)
    locked: bool = False  # kept when the plan is rebuilt
    score: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.src_in_ms >= self.src_out_ms:
            raise ValueError("src_in_ms must be < src_out_ms")
        return self


class SourceAudio(Strict):
    mode: Literal["mute", "duck", "full"] = "duck"
    stem: Literal["mix", "no_vocals", "vocals"] = "mix"
    gain_db: float = 0.0


class PlanSegment(Strict):
    id: SegmentId
    kind: Literal["narration", "original"]
    text: str | None = None  # the narration text the audio was spoken from (ADR-0001)
    text_hash: Sha256 | None = None  # hash of the script text this was built from
    stale: bool = False
    voice: VoiceSpec | None = None
    voice_pinned: bool = False  # a person chose the voice / speed: a rebuild keeps it
    line_refs: list[LineId] = []  # original: the transcript lines played (a rebuild compares)
    audio: AudioRef | None = None
    clips: list[Clip] = []
    source_audio: SourceAudio = SourceAudio()

    @model_validator(mode="after")
    def _kind_rules(self) -> Self:
        if self.text is not None and self.text_hash is not None:
            digest = "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if digest != self.text_hash:
                raise ValueError("text_hash does not match text")
        if self.kind == "narration":
            if self.voice is None:
                raise ValueError("narration segment needs a voice")
            if not self.stale and not (self.text and self.text.strip()):
                raise ValueError("non-stale narration segment needs its text")
            if not self.stale and (self.audio is None or not self.clips):
                raise ValueError("non-stale narration segment needs audio and clips")
        else:
            if not self.clips:
                raise ValueError("original segment needs clips")
            if self.audio is not None:
                raise ValueError("original segment has no narration audio")
        return self


class Bgm(Strict):
    file: str
    gain_db: float = -22.0
    duck_under_narration_db: float = -8.0


class EditPlan(Versioned):
    id: PlanId
    project_id: ProjectId
    version: int = Field(ge=1)
    parent_version: int | None = None
    author: Author
    script_ref: ScriptRef
    segments: list[PlanSegment]
    bgm: Bgm | None = None
    output_profile: str = "source"

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment ids")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent_version must be < version")
        return self
