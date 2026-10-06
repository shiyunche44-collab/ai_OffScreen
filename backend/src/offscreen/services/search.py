"""Searching a movie's shots by what they show: `search_shots(text, filters)`.

The query is embedded twice - into the picture space (compared with the keyframes) and as text
(compared with the shot descriptions) - and the two rankings are merged by reciprocal rank
fusion (`algo.search.fuse`). The index is derived from the `analysis.embeddings` artifact the
first time it is needed (and rebuilt if the artifact changed), under `data/index/`; losing it
costs nothing."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from offscreen.algo.search import fuse
from offscreen.config import AppConfig
from offscreen.domain.index import ShotIndex, Shots
from offscreen.engine import Artifact
from offscreen.providers.adapters.lancedb_index import LanceShotIndex
from offscreen.providers.ports import Embedder, IndexedShot, ShotFilter, VectorIndex
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.embeddings import (
    IMAGE_VECTORS_FILE,
    SHOT_INDEX_FILE,
    TEXT_VECTORS_FILE,
)
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo

MAX_LIMIT = 100
POOL = 3
"""Each ranking is read this many times deeper than the answer, so the merge has something to
choose from."""


class ShotHit(BaseModel):
    shot_id: str
    start_ms: int
    end_ms: int
    score: float
    """Rank fusion score; only meaningful for ordering."""
    similarity: dict[str, float]
    """Cosine similarity per ranking the shot appeared in (`image`, `text`)."""
    scene_id: str | None = None
    caption: str = ""
    thumbnail: str | None = None
    """The shot's middle keyframe, relative to the data directory."""


class ShotSearchResult(BaseModel):
    query: str
    rankings: list[str] = Field(description="Which rankings were used: image, text.")
    hits: list[ShotHit]


class SearchService:
    def __init__(
        self,
        cfg: AppConfig,
        db: Database,
        jobs: JobService,
        index: VectorIndex | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)
        self.index = index or LanceShotIndex()
        self._lock = threading.Lock()

    def search(
        self,
        asset_id: str,
        query: str,
        *,
        limit: int = 10,
        where: ShotFilter | None = None,
    ) -> ShotSearchResult:
        text = query.strip()
        if not text:
            raise InvalidInput("the query is empty")
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidInput(f"limit must be between 1 and {MAX_LIMIT}")
        if self.assets.get(asset_id) is None:
            raise NotFound(f"unknown asset {asset_id}")
        where = where or ShotFilter()

        providers = self.jobs.providers
        with Pipeline(self.cfg, providers, db=self.db) as p:
            art = p.peek("analysis.embeddings", asset_id)
            frames = p.peek("analysis.keyframes", asset_id)
        if art is None or frames is None:
            raise NotFound("analysis.embeddings has not been built yet")
        meta = art.read_model(SHOT_INDEX_FILE, ShotIndex)
        path = self._ensure_index(asset_id, art, meta)

        rankings: dict[str, list[str]] = {}
        similarity: dict[str, dict[str, float]] = {}
        wanted = limit * POOL
        columns: list[tuple[Literal["image", "text"], Embedder | None, bool]] = [
            ("image", providers.image_embedder, meta.image_model is not None),
            ("text", providers.text_embedder, meta.text_model is not None),
        ]
        for column, embedder, has in columns:
            if not has or embedder is None:
                continue
            (vector,) = embedder.embed_texts([text])
            hits = self.index.search(
                path,
                column,
                vector,
                limit=wanted,
                where=where,
            )
            rankings[column] = [h.shot_id for h in hits]
            for h in hits:
                similarity.setdefault(h.shot_id, {})[column] = round(h.similarity, 4)
        if not rankings:
            raise InvalidInput("no embedder is configured for searching")

        shots = {s.id: s for s in frames.read_model(SHOTS_FILE, Shots).shots}
        by_id = {e.shot_id: e for e in meta.shots}
        base = self.cfg.data_dir.resolve()
        out: list[ShotHit] = []
        for shot_id, score in fuse(rankings)[:limit]:
            entry = by_id[shot_id]
            shot = shots.get(shot_id)
            out.append(
                ShotHit(
                    shot_id=shot_id,
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    score=round(score, 6),
                    similarity=similarity[shot_id],
                    scene_id=entry.scene_id,
                    caption=entry.caption,
                    thumbnail=frames.path(shot.keyframes[1]).resolve().relative_to(base).as_posix()
                    if shot and len(shot.keyframes) > 1
                    else None,
                )
            )
        return ShotSearchResult(query=text, rankings=list(rankings), hits=out)

    # ---- the derived index ---------------------------------------------------------------
    def _ensure_index(self, asset_id: str, art: Artifact, meta: ShotIndex) -> Path:
        path = self.cfg.data_dir / "index" / asset_id / art.cache_key.split(":")[-1][:16]
        with self._lock:
            if (path / ".complete").is_file():
                return path
            image = _load(art, IMAGE_VECTORS_FILE) if meta.image_model else None
            text = _load(art, TEXT_VECTORS_FILE) if meta.text_model else None
            rows = [
                IndexedShot(
                    shot_id=e.shot_id,
                    scene_id=e.scene_id,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    sharpness=e.sharpness,
                    brightness=e.brightness,
                    is_credits=e.is_credits,
                    image=image[i] if image is not None else None,
                    text=text[i] if text is not None else None,
                )
                for i, e in enumerate(meta.shots)
            ]
            self.index.build(path, rows)
            (path / ".complete").write_text("ok")
            self._drop_old(path.parent, keep=path)
            return path

    @staticmethod
    def _drop_old(directory: Path, keep: Path) -> None:
        import shutil

        for other in directory.iterdir():
            if other != keep and other.is_dir():
                shutil.rmtree(other, ignore_errors=True)


def _load(art: Artifact, name: str) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.load(art.path(name))]
