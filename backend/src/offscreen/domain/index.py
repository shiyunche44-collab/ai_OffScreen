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
    Rational,
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
    keyframes: list[str] = []  # relative to the artifact directory holding this document
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


class SpriteTile(Strict):
    shot_id: ShotId
    sheet: int = Field(ge=0)
    """Index into `SpriteSheets.sheets`."""
    col: int = Field(ge=0)
    row: int = Field(ge=0)


class SpriteSheets(Versioned):
    """Thumbnails of every shot (one frame each) packed into a few large images, so the shot strip
    of the analysis page loads a handful of files instead of one request per shot. A tile sits at
    `(col * tile_width, row * tile_height)` of its sheet."""

    asset_id: AssetId
    tile_width: int = Field(gt=0)
    tile_height: int = Field(gt=0)
    columns: int = Field(gt=0)
    rows: int = Field(gt=0)
    sheets: list[str]  # relative to the artifact directory holding this document
    tiles: list[SpriteTile]

    @model_validator(mode="after")
    def _tiles_fit(self) -> Self:
        for t in self.tiles:
            if t.sheet >= len(self.sheets) or t.col >= self.columns or t.row >= self.rows:
                raise ValueError(f"tile of {t.shot_id} lies outside the sheets")
        ids = [t.shot_id for t in self.tiles]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shot ids in tiles")
        return self


class ShotSignature(Strict):
    shot_id: ShotId
    frames: list[list[int]]
    """One colour histogram per keyframe (same order as `Shot.keyframes`), in per-mille: bin
    `r * n * n + g * n + b` (n = bins_per_channel) holds the share of pixels (sums to ~1000)."""


class VisualSignatures(Versioned):
    """A compact colour fingerprint of every keyframe, so later stages can tell how much the
    picture changes from one shot to the next without decoding images again."""

    asset_id: AssetId
    bins_per_channel: int = Field(gt=0)
    signatures: list[ShotSignature]

    @model_validator(mode="after")
    def _bins_match(self) -> Self:
        want = self.bins_per_channel**3
        for sig in self.signatures:
            if any(len(h) != want or min(h, default=0) < 0 for h in sig.frames):
                raise ValueError(f"signature of {sig.shot_id} does not have {want} bins")
        ids = [x.shot_id for x in self.signatures]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shot ids in signatures")
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
    frame: int = Field(default=0, ge=0)
    """Which of the shot's keyframes the face was found in."""
    score: Unit | None = None
    """The detector's confidence."""
    embedding: int | None = Field(default=None, ge=0)
    """Row of this face in `embeddings.npy` next to the document (the file is the `Faces`
    artifact's second output; rows follow the order of the faces in the document)."""


class ShotFaces(Strict):
    shot_id: ShotId
    faces: list[FaceBox]


class Faces(Versioned):
    asset_id: AssetId
    shots: list[ShotFaces]


class CastMember(Strict):
    character_id: CharacterId
    share: Unit
    """Share of the shot's identified face area (the members of a shot sum to 1)."""
    area_ratio: Unit
    """The character's largest face in the shot, as a share of the frame."""


class ShotCast(Strict):
    shot_id: ShotId
    characters: list[CastMember] = []
    """Largest share first; empty when nobody is identified."""


class Cast(Versioned):
    """Who is on screen in each shot: what scene summaries and footage matching read, so they
    need not know about faces."""

    asset_id: AssetId
    shots: list[ShotCast]


class ShotIndexEntry(Strict):
    shot_id: ShotId
    scene_id: SceneId | None = None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    sharpness: Unit = 0.0
    brightness: Unit = 0.0
    is_credits: bool = False
    caption: str = ""
    """The description the text vector was made from (what a result shows)."""


class ShotIndex(Versioned):
    """What the search over shots reads: one entry per shot, in shot order. The vectors are the
    rows of `image_vectors.npy` / `text_vectors.npy` next to the document (float32, unit length,
    same order); a file is absent when that embedder was not configured."""

    asset_id: AssetId
    image_model: str | None = None
    text_model: str | None = None
    shots: list[ShotIndexEntry]


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
    centroid: list[float] | None = None
    """The character's face centre when the edit was made. After a re-clustering the ids change;
    this is what finds the same person again."""
    merged_into_centroid: list[float] | None = None
    """Likewise for `merged_into`."""


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
    ending_scene_ids: list[SceneId] = []
    """The scenes the ending is drawn from (a claim about the plot, so it has its anchors too)."""
    themes: list[str] = []


class CutAnnotations(Versioned):
    """Where a person says the hard cuts of a movie are: the ground truth shot detection is
    measured against. Human work, kept apart from every analysis output."""

    asset_id: AssetId
    fps: Rational
    """The frame rate `cuts` is counted in (the movie's own)."""
    cuts: list[int] = []
    """Frame numbers (from 0) of the first frame of each new shot; ascending, no duplicates, and
    never 0 (the film starts a shot by itself)."""

    @model_validator(mode="after")
    def _ascending(self) -> Self:
        if any(c < 1 for c in self.cuts):
            raise ValueError("a cut is the first frame of a new shot, so it is at least 1")
        if self.cuts != sorted(set(self.cuts)):
            raise ValueError("cuts must be ascending without duplicates")
        return self


class SelectionLabel(Strict):
    """One piece of narration with the shots a person says would be fine to show under it."""

    id: str = Field(pattern=r"^[0-9A-Za-z_-]+$")
    text: str = Field(min_length=1)
    """The narration, as it would be spoken."""
    scene_refs: list[SceneId] = Field(min_length=1)
    """The scenes the text is about (a narration segment always cites some)."""
    acceptable: list[ShotId] = Field(min_length=1)
    """Every shot that would be an acceptable picture for the text, not just the best one."""
    note: str | None = None


class SelectionAnnotations(Versioned):
    """Ground truth for footage selection: what the ranking of shots is measured against. Human
    work, kept apart from every analysis output."""

    asset_id: AssetId
    labels: list[SelectionLabel] = []

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [x.id for x in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate label ids")
        for label in self.labels:
            if len(label.acceptable) != len(set(label.acceptable)):
                raise ValueError(f"{label.id}: duplicate acceptable shots")
        return self
