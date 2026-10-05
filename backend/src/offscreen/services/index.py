"""Reading a movie's MovieIndex (L1) for display: transcript, shots, scenes, story.

Read-only and cheap: it peeks at the artifact cache and runs nothing. Files the front end needs
(proxy video, keyframes, sprite sheets) are returned as paths relative to the data directory, to
be fetched from `/api/files/...`."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from offscreen.config import AppConfig
from offscreen.domain.index import (
    Captions,
    Scenes,
    Shot,
    ShotCaption,
    ShotQuality,
    Shots,
    SpriteSheets,
    Story,
    Transcript,
)
from offscreen.engine import Artifact
from offscreen.services.errors import NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.keyframes import SPRITES_FILE
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo

M = TypeVar("M", bound=BaseModel)


class SpriteSlot(BaseModel):
    sheet: int
    """Index into `ShotsView.sheets`."""
    col: int
    row: int


class ShotView(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    quality: ShotQuality | None = None
    keyframes: list[str]
    """Three frames (10 / 50 / 90 %), relative to the data directory."""
    caption: ShotCaption | None = None
    sprite: SpriteSlot | None = None


class ShotsView(BaseModel):
    asset_id: str
    video: str
    """The 540p proxy to play, relative to the data directory."""
    tile_width: int
    tile_height: int
    columns: int
    rows: int
    sheets: list[str]
    """Sprite sheets, relative to the data directory. A shot's thumbnail sits at
    `(col * tile_width, row * tile_height)` of its sheet."""
    shots: list[ShotView]


class IndexService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)

    def transcript(self, asset_id: str) -> Transcript:
        return self._read(asset_id, "analysis.transcript", TRANSCRIPT_FILE, Transcript)

    def scenes(self, asset_id: str) -> Scenes:
        return self._read(asset_id, "analysis.scenes", SCENES_FILE, Scenes)

    def story(self, asset_id: str) -> Story:
        return self._read(asset_id, "analysis.story", STORY_FILE, Story)

    def shots(self, asset_id: str) -> ShotsView:
        self._require_asset(asset_id)
        with self._pipeline() as p:
            frames = self._built(p, "analysis.keyframes", asset_id)
            proxy = self._built(p, "analysis.proxy", asset_id)
            captured = p.peek("analysis.captions", asset_id)
            shots = frames.read_model(SHOTS_FILE, Shots).shots
            sprites = frames.read_model(SPRITES_FILE, SpriteSheets)
            captions = (
                {c.shot_id: c for c in captured.read_model(CAPTIONS_FILE, Captions).captions}
                if captured is not None
                else {}
            )
            slots = {
                t.shot_id: SpriteSlot(sheet=t.sheet, col=t.col, row=t.row) for t in sprites.tiles
            }
            return ShotsView(
                asset_id=asset_id,
                video=self._rel(proxy, PROXY_FILE),
                tile_width=sprites.tile_width,
                tile_height=sprites.tile_height,
                columns=sprites.columns,
                rows=sprites.rows,
                sheets=[self._rel(frames, s) for s in sprites.sheets],
                shots=[self._shot(frames, s, captions, slots) for s in shots],
            )

    # ---- internals -------------------------------------------------------------------------
    def _shot(
        self,
        frames: Artifact,
        shot: Shot,
        captions: dict[str, ShotCaption],
        slots: dict[str, SpriteSlot],
    ) -> ShotView:
        return ShotView(
            id=shot.id,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            quality=shot.quality,
            keyframes=[self._rel(frames, k) for k in shot.keyframes],
            caption=captions.get(shot.id),
            sprite=slots.get(shot.id),
        )

    def _read(self, asset_id: str, stage: str, file: str, cls: type[M]) -> M:
        self._require_asset(asset_id)
        with self._pipeline() as p:
            return self._built(p, stage, asset_id).read_model(file, cls)

    @staticmethod
    def _built(p: Pipeline, stage: str, asset_id: str) -> Artifact:
        artifact = p.peek(stage, asset_id)
        if artifact is None:
            raise NotFound(f"{stage} has not been built yet")
        return artifact

    def _rel(self, artifact: Artifact, rel: str) -> str:
        return artifact.path(rel).resolve().relative_to(self.cfg.data_dir.resolve()).as_posix()

    def _require_asset(self, asset_id: str) -> None:
        if self.assets.get(asset_id) is None:
            raise NotFound(f"unknown asset {asset_id}")

    def _pipeline(self) -> Pipeline:
        return Pipeline(self.cfg, self.jobs.providers, db=self.db)
