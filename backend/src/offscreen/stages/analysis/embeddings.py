"""embeddings: keyframes + descriptions -> vectors to search shots by (ARCHITECTURE §7.1).

The middle keyframe of every shot goes through the image embedder (a joint image-text model, so a
sentence can later be compared with it); the shot's description (what is seen, the action, the
mood) goes through the caption embedder as text. Either may be absent; at least one must be
configured. Outputs: `shot_index.json` (what to filter and show, one entry per shot in order) and
the vector files `image_vectors.npy` / `text_vectors.npy` (float32, unit length, same order).
The searchable index itself is derived from these when first needed (services.search)."""

from __future__ import annotations

from typing import Any

import numpy as np

from offscreen.algo.search import search_text
from offscreen.domain.index import (
    Captions,
    Scenes,
    ShotIndex,
    ShotIndexEntry,
    Shots,
)
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.ports import Embedder
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

SHOT_INDEX_FILE = "shot_index.json"
IMAGE_VECTORS_FILE = "image_vectors.npy"
TEXT_VECTORS_FILE = "text_vectors.npy"
BATCH = 32
MIDDLE = 1
"""Which keyframe of a shot is embedded (the 50 % one)."""


class EmbeddingsError(RuntimeError):
    pass


class EmbeddingsStage(Stage):
    name = "analysis.embeddings"
    version = 1
    lane: Lane = "gpu"

    def __init__(self, images: Embedder | None, texts: Embedder | None) -> None:
        self.images = images
        """Embeds keyframes (None: no picture search)."""
        self.texts = texts
        """Embeds descriptions (None: no text search)."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [
            ArtifactRef("analysis.keyframes", scope),
            ArtifactRef("analysis.captions", scope),
            ArtifactRef("analysis.scenes", scope),
        ]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"asset_id": scope["asset_id"], "keyframe": MIDDLE}

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "image_embedder": self.images.id if self.images else "none",
            "text_embedder": self.texts.id if self.texts else "none",
        }

    def run(self, ctx: StageContext) -> StageOutput:
        if self.images is None and self.texts is None:
            raise EmbeddingsError("no embedder is configured")
        frames = ctx.input("analysis.keyframes")
        shots = frames.read_model(SHOTS_FILE, Shots)
        captions = {
            c.shot_id: c
            for c in ctx.input("analysis.captions").read_model(CAPTIONS_FILE, Captions).captions
        }
        scene_of = {
            shot_id: scene.id
            for scene in ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
            for shot_id in scene.shot_ids
        }
        if not shots.shots:
            raise EmbeddingsError("there are no shots to embed")

        entries = [
            ShotIndexEntry(
                shot_id=s.id,
                scene_id=scene_of.get(s.id),
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                sharpness=s.quality.sharpness if s.quality else 0.0,
                brightness=s.quality.brightness if s.quality else 0.0,
                is_credits=bool(captions[s.id].is_credits) if s.id in captions else False,
                caption=search_text(captions[s.id]) if s.id in captions else "",
            )
            for s in shots.shots
        ]

        image_dim: int | None = None
        if self.images is not None:
            paths = [frames.path(s.keyframes[MIDDLE]) for s in shots.shots]
            rows: list[list[float]] = []
            for start in range(0, len(paths), BATCH):
                if ctx.is_canceled():
                    raise StageCanceled(self.name)
                rows += self.images.embed_images(paths[start : start + BATCH])
                ctx.progress(0.7 * len(rows) / len(paths), f"{len(rows)}/{len(paths)} frames")
            image_dim = self._save(ctx, IMAGE_VECTORS_FILE, rows, len(paths))

        text_dim: int | None = None
        if self.texts is not None:
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            rows = []
            texts = [e.caption for e in entries]
            for start in range(0, len(texts), BATCH * 4):
                rows += self.texts.embed_texts(texts[start : start + BATCH * 4])
            text_dim = self._save(ctx, TEXT_VECTORS_FILE, rows, len(texts))

        write_model(
            ctx.out_dir / SHOT_INDEX_FILE,
            ShotIndex(
                asset_id=shots.asset_id,
                image_model=self.images.id if self.images else None,
                text_model=self.texts.id if self.texts else None,
                shots=entries,
            ),
        )
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={"shots": len(entries), "image_dim": image_dim, "text_dim": text_dim}
        )

    @staticmethod
    def _save(ctx: StageContext, name: str, rows: list[list[float]], expected: int) -> int:
        if len(rows) != expected:
            raise EmbeddingsError(f"{name}: got {len(rows)} vectors for {expected} inputs")
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise EmbeddingsError(f"{name}: the embedder returned unusable vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0, 1.0, norms)
        np.save(ctx.out_dir / name, matrix.astype(np.float32))
        return int(matrix.shape[1])
