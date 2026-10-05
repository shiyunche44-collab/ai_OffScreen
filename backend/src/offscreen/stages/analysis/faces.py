"""faces: keyframes -> where the faces are and what they look like (ARCHITECTURE §7.1).

Every keyframe of every shot goes through the `FaceAnalyzer` port (never frame by frame of the
film). Faces that are too unsure or too small to tell apart are dropped. Output: `faces.json`
(boxes per shot) and `embeddings.npy` (one unit-length row per face, in document order), which
the clustering step (M3-08) reads."""

from __future__ import annotations

from typing import Any

import numpy as np

from offscreen.algo.faces import box_area, unit_vector
from offscreen.domain.index import FaceBox, Faces, ShotFaces, Shots
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.ports import DetectedFace, FaceAnalyzer
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

FACES_FILE = "faces.json"
EMBEDDINGS_FILE = "embeddings.npy"
MIN_SCORE = 0.5
"""Detections less sure than this are not faces."""
MIN_AREA_RATIO = 0.004
"""Smaller than this (of the frame) there is too little to recognise a person by."""


class FacesError(RuntimeError):
    pass


class FacesStage(Stage):
    name = "analysis.faces"
    version = 1
    lane: Lane = "gpu"

    def __init__(self, analyzer: FaceAnalyzer | None) -> None:
        self.analyzer = analyzer
        """None: no face analysis is set up; running the stage says so."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.keyframes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "min_score": MIN_SCORE,
            "min_area_ratio": MIN_AREA_RATIO,
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {"faces": self.analyzer.id if self.analyzer else "none"}

    def run(self, ctx: StageContext) -> StageOutput:
        if self.analyzer is None:
            raise FacesError("no face analyzer is configured")
        frames = ctx.input("analysis.keyframes")
        doc = frames.read_model(SHOTS_FILE, Shots)

        shots: list[ShotFaces] = []
        rows: list[list[float]] = []
        for i, shot in enumerate(doc.shots):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            boxes: list[FaceBox] = []
            for frame, rel in enumerate(shot.keyframes):
                found = self.analyzer.detect_and_embed(frames.path(rel))
                for face in sorted(found, key=lambda f: -box_area(f.bbox)):
                    box = self._box(face, frame, len(rows))
                    vector = unit_vector(face.embedding) if box is not None else None
                    if box is None or vector is None:
                        continue
                    if rows and len(vector) != len(rows[0]):
                        raise FacesError(
                            f"{self.analyzer.id} returned features of different lengths "
                            f"({len(vector)} and {len(rows[0])})"
                        )
                    rows.append(vector)
                    boxes.append(box)
            shots.append(ShotFaces(shot_id=shot.id, faces=boxes))
            ctx.progress((i + 1) / len(doc.shots), shot.id)

        write_model(ctx.out_dir / FACES_FILE, Faces(asset_id=doc.asset_id, shots=shots))
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(rows[0]) if rows else 0)
        np.save(ctx.out_dir / EMBEDDINGS_FILE, matrix)
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "shots": len(shots),
                "shots_with_faces": sum(1 for s in shots if s.faces),
                "faces": len(rows),
            }
        )

    @staticmethod
    def _box(face: DetectedFace, frame: int, row: int) -> FaceBox | None:
        area = box_area(face.bbox)
        if face.score < MIN_SCORE or area < MIN_AREA_RATIO:
            return None
        x0, y0, x1, y1 = face.bbox
        return FaceBox(
            bbox=(x0, y0, x1, y1),
            area_ratio=min(1.0, area),
            frame=frame,
            score=face.score,
            embedding=row,
        )
