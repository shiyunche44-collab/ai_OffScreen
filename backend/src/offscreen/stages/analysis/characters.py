"""characters: faces -> people (ARCHITECTURE §7.1).

The face features are grouped by who they belong to (`algo.clustering`); every group big enough
becomes a character (`ch_01` is the one seen most), and the faces get their character id. Names
come later (M3-09) and are never invented here. Outputs:

- `characters.json`: the characters, with the face count, a pointer to the centre
  (`centroid_ref` = `centroids.npy#<row>`, so a later re-clustering can map human edits back) and
  up to three thumbnails (`faces/<id>_<n>.jpg`, cropped from the keyframes, distinct shots)
- `faces.json`: the input document with `character_id` filled in
- `cast.json`: per shot, who is on screen and how much of the face area they take
- `centroids.npy`: one unit-length row per character"""

from __future__ import annotations

from typing import Any

import numpy as np

from offscreen.algo.clustering import (
    MERGE_SIMILARITY,
    MICRO_SIMILARITY,
    MIN_FACES,
    MIN_SHOTS,
    NOISE,
    centroids,
    cluster_faces,
    shot_cast,
    shot_indexes,
)
from offscreen.domain.index import (
    Cast,
    CastMember,
    Character,
    Characters,
    FaceBox,
    FaceCluster,
    Faces,
    ShotCast,
    ShotFaces,
    Shots,
)
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.transcode import crop_image
from offscreen.stages.analysis.faces import EMBEDDINGS_FILE, FACES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

CHARACTERS_FILE = "characters.json"
CAST_FILE = "cast.json"
CENTROIDS_FILE = "centroids.npy"
THUMBNAILS = 3


class CharactersError(RuntimeError):
    pass


def character_id(rank: int) -> str:
    """`ch_01` for the character seen most, then `ch_02`, ..."""
    return f"ch_{rank + 1:02d}"


class CharactersStage(Stage):
    name = "analysis.characters"
    version = 1
    lane: Lane = "cpu"

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.faces", scope), ArtifactRef("analysis.keyframes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "micro": MICRO_SIMILARITY,
            "merge": MERGE_SIMILARITY,
            "min_faces": MIN_FACES,
            "min_shots": MIN_SHOTS,
            "thumbnails": THUMBNAILS,
        }

    def run(self, ctx: StageContext) -> StageOutput:
        faces_art = ctx.input("analysis.faces")
        frames = ctx.input("analysis.keyframes")
        doc = faces_art.read_model(FACES_FILE, Faces)
        shots = {s.id: s for s in frames.read_model(SHOTS_FILE, Shots).shots}
        vectors = np.load(faces_art.path(EMBEDDINGS_FILE))
        flat = [f for s in doc.shots for f in s.faces]
        if len(vectors) != len(flat):
            raise CharactersError(
                f"faces.json lists {len(flat)} faces but embeddings.npy has {len(vectors)} rows"
            )

        labels = cluster_faces(vectors, shot_indexes(doc.shots))
        ctx.progress(0.3, "grouped")

        named: list[ShotFaces] = []
        position = 0
        for s in doc.shots:
            boxes: list[FaceBox] = []
            for f in s.faces:
                label = int(labels[position])
                position += 1
                boxes.append(
                    f.model_copy(
                        update={"character_id": None if label == NOISE else character_id(label)}
                    )
                )
            named.append(ShotFaces(shot_id=s.shot_id, faces=boxes))

        count = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
        centres = centroids(np.asarray(vectors, dtype=np.float64), labels)
        np.save(ctx.out_dir / CENTROIDS_FILE, centres.astype(np.float32))
        (ctx.out_dir / "faces").mkdir()

        characters: list[Character] = []
        try:
            for rank in range(count):
                mine = [
                    (s.shot_id, f)
                    for s in named
                    for f in s.faces
                    if f.character_id == character_id(rank)
                ]
                thumbs = self._thumbnails(ctx, frames, shots, character_id(rank), mine)
                characters.append(
                    Character(
                        id=character_id(rank),
                        face_cluster=FaceCluster(
                            size=len(mine),
                            centroid_ref=f"{CENTROIDS_FILE}#{rank}",
                            thumbnails=thumbs,
                        ),
                    )
                )
                ctx.progress(0.3 + 0.7 * (rank + 1) / count, character_id(rank))
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e

        write_model(
            ctx.out_dir / CHARACTERS_FILE, Characters(asset_id=doc.asset_id, characters=characters)
        )
        write_model(ctx.out_dir / FACES_FILE, Faces(asset_id=doc.asset_id, shots=named))
        write_model(
            ctx.out_dir / CAST_FILE,
            Cast(
                asset_id=doc.asset_id,
                shots=[
                    ShotCast(
                        shot_id=s.shot_id,
                        characters=[
                            CastMember(character_id=c, share=share, area_ratio=largest)
                            for c, share, largest in shot_cast(s.faces)
                        ],
                    )
                    for s in named
                ],
            ),
        )
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "faces": len(flat),
                "characters": count,
                "unassigned_faces": int((labels == NOISE).sum()),
            }
        )

    @staticmethod
    def _thumbnails(
        ctx: StageContext,
        frames: Any,
        shots: dict[str, Any],
        cid: str,
        mine: list[tuple[str, FaceBox]],
    ) -> list[str]:
        """The clearest faces of the character, at most one per shot, cropped from the keyframe."""
        best = sorted(mine, key=lambda m: -((m[1].score or 0.0) * m[1].area_ratio ** 0.5))
        picked: list[tuple[str, FaceBox]] = []
        for shot_id, f in best:
            if all(shot_id != p[0] for p in picked):
                picked.append((shot_id, f))
            if len(picked) == THUMBNAILS:
                break
        out: list[str] = []
        for n, (shot_id, f) in enumerate(picked, start=1):
            rel = f"faces/{cid}_{n}.jpg"
            source = frames.path(shots[shot_id].keyframes[f.frame])
            crop_image(source, ctx.out_dir / rel, f.bbox, should_cancel=ctx.is_canceled)
            out.append(rel)
        return out
