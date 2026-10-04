"""L1 MovieIndex: everything the analysis stages learn about a movie.

Each analysis stage writes its own document; `characters.overrides` is the human
revision layer and is never merged into the AI output on disk."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import (
    AssetId,
    CharacterId,
    LineId,
    SceneId,
    ShotId,
    Strict,
    TimeRange,
    Versioned,
)

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


# ---- transcript -----------------------------------------------------------
class Word(TimeRange):
    w: str


class TranscriptLine(TimeRange):
    id: LineId
    speaker: str | None = None
    text: str
    words: list[Word] = []


class Transcript(Versioned):
    asset_id: AssetId
    language: str
    source: str  # e.g. "asr:faster-whisper/large-v3" or "subtitle:external"
    lines: list[TranscriptLine]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [x.id for x in self.lines]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate line ids")
        return self


# ---- shots, captions, faces ------------------------------------------------
class ShotQuality(Strict):
    sharpness: Unit
    brightness: Unit


class Shot(TimeRange):
    id: ShotId
    keyframes: list[str] = []  # relative to data_dir
    quality: ShotQuality | None = None


class Shots(Versioned):
    asset_id: AssetId
    shots: list[Shot]

    @model_validator(mode="after")
    def _ordered_unique(self) -> Self:
        ids = [s.id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shot ids")
        for a, b in zip(self.shots, self.shots[1:], strict=False):
            if b.start_ms < a.end_ms:
                raise ValueError(f"shots overlap or are unordered: {a.id}, {b.id}")
        return self


ShotSize = Literal["extreme_close_up", "close_up", "medium", "wide", "extreme_wide", "other"]


class ShotCaption(Strict):
    shot_id: ShotId
    caption: str
    shot_size: ShotSize = "other"
    action: str | None = None
    emotion: str | None = None
    has_onscreen_text: bool = False
    is_credits: bool = False


class Captions(Versioned):
    asset_id: AssetId
    captions: list[ShotCaption]


class FaceBox(Strict):
    character_id: CharacterId | None = None
    bbox: tuple[Unit, Unit, Unit, Unit]  # x0, y0, x1, y1, normalized
    area_ratio: Unit


class ShotFaces(Strict):
    shot_id: ShotId
    faces: list[FaceBox]


class Faces(Versioned):
    asset_id: AssetId
    shots: list[ShotFaces]


# ---- scenes, characters, story ----------------------------------------------
class Scene(TimeRange):
    id: SceneId
    shot_ids: list[ShotId] = Field(min_length=1)
    line_ids: list[LineId] = []
    summary: str
    characters: list[CharacterId] = []
    location: str | None = None
    importance: Unit = 0.5


class Scenes(Versioned):
    asset_id: AssetId
    scenes: list[Scene]

    @model_validator(mode="after")
    def _ordered_unique(self) -> Self:
        ids = [s.id for s in self.scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scene ids")
        for a, b in zip(self.scenes, self.scenes[1:], strict=False):
            if b.start_ms < a.end_ms:
                raise ValueError(f"scenes overlap or are unordered: {a.id}, {b.id}")
        return self


class FaceCluster(Strict):
    size: int = Field(ge=1)
    centroid_ref: str | None = None
    thumbnails: list[str] = []


class Character(Strict):
    id: CharacterId
    name: str | None = None
    name_source: Literal["ai", "human", "tmdb"] | None = None
    aliases: list[str] = []
    role: str | None = None
    bio: str | None = None
    face_cluster: FaceCluster | None = None


class Characters(Versioned):
    asset_id: AssetId
    characters: list[Character]


class CharacterOverride(Strict):
    character_id: CharacterId
    name: str | None = None
    aliases: list[str] | None = None
    ignored: bool = False
    merged_into: CharacterId | None = None


class CharacterOverrides(Versioned):
    asset_id: AssetId
    overrides: list[CharacterOverride] = []


class Act(Strict):
    name: str
    scene_ids: list[SceneId] = Field(min_length=1)
    summary: str


class TurningPoint(Strict):
    scene_id: SceneId
    what: str


class Relation(Strict):
    a: CharacterId
    b: CharacterId
    relation: str


class Story(Versioned):
    """Every claim about the plot carries scene ids: the anchor against hallucination."""

    asset_id: AssetId
    logline: str
    synopsis: str
    acts: list[Act]
    turning_points: list[TurningPoint] = []
    relations: list[Relation] = []
    ending: str | None = None
    themes: list[str] = []
